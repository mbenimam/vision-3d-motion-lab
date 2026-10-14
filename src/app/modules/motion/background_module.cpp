// Topic: background subtraction with a static camera (mean, approximate median, Σ-Δ, Gaussian),
// with morphological or Markovian regularisation of the foreground mask. Baseline: OpenCV's
// mixture-of-Gaussians (MOG2) or KNN subtractor on the same frames, from the same start.

#include <algorithm>
#include <cmath>
#include <optional>

#include <opencv2/imgproc.hpp>

#include "app/module.hpp"
#include "app/modules/motion/motion_samples.hpp"
#include "core/common/image.hpp"
#include "core/motion/background.hpp"
#include "core/motion/opencv_baselines.hpp"
#include "core/motion/regularize.hpp"

namespace vl::app {
namespace {

using namespace vl::motion;

constexpr double kDiffMax = 48.0;  // top of the colour scale for |I − B| and T, grey levels

enum class Reg { None, Morphology, Mrf };

// Dimmed grey copy of the frame, so coloured masks read clearly on top of it.
cv::Mat dimmed(const cv::Mat& bgr) {
    cv::Mat g, out;
    cv::cvtColor(bgr, g, cv::COLOR_BGR2GRAY);
    cv::cvtColor(g * 0.45, out, cv::COLOR_GRAY2BGR);
    return out;
}

cv::Scalar bgr(const QColor& c, double k = 1.0) { return {k * c.blue(), k * c.green(), k * c.red()}; }

// Outline of a mask as closed polylines (pixel corners), for the ground truth.
std::vector<Shape> outline(const cv::Mat& mask, const QColor& color) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<Shape> shapes;
    for (const auto& c : contours) {
        QPolygonF poly;
        for (const cv::Point& p : c) poly << QPointF(p.x + 0.5, p.y + 0.5);
        shapes.push_back(PolylineShape{poly, Pen{color, 1.5, false}, true});
    }
    return shapes;
}

double mean(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / double(v.size());
}

// Per-run measurements of one detector (ours or OpenCV's).
struct Scores {
    std::vector<double> f1, precision, recall, share;
    double ms = 0;  // time spent in the method itself (model + regularisation), all frames
    void add(const cv::Mat& fg, const cv::Mat* truth) {
        share.push_back(cv::countNonZero(fg) / double(fg.total()));
        if (!truth) return;
        const Confusion c = compareMasks(fg, *truth);
        f1.push_back(c.f1());
        precision.push_back(c.precision());
        recall.push_back(c.recall());
    }
};

class BackgroundModule final : public Module {
public:
    Info info() const override {
        return {QStringLiteral("background"), QStringLiteral("Background subtraction"), QStringLiteral("Motion"),
                QStringLiteral("With a camera that does not move, every pixel keeps a running estimate of the empty "
                               "scene, updated a little at each frame; a pixel too far from its estimate is "
                               "<i>foreground</i>. <b>Σ-Δ</b> moves the estimate by ±1 grey level per frame and learns, "
                               "the same way, how much each pixel usually fluctuates, so the threshold adapts per "
                               "pixel. A Markov field then cleans the mask: a label pays for each neighbour that "
                               "disagrees, so isolated noise flips back and holes fill."),
                QStringLiteral("on <i>Synthetic: orange disc</i>, set Regularisation to <b>None</b> and watch speckles "
                               "pull F1 down; then switch off <b>Start from a temporal median</b> to see the ghost of "
                               "frame 0 fade at ±1 per frame."),
                InputKind::Video, 500, QStringLiteral("Foreground"),
                QStringLiteral("cv::BackgroundSubtractorMOG2 / KNN"),
                {QStringLiteral("src/core/motion/background.hpp"), QStringLiteral("src/core/motion/background.cpp"),
                 QStringLiteral("src/core/motion/regularize.hpp"), QStringLiteral("src/core/motion/regularize.cpp")}};
    }

    std::vector<ParamSpec> params() const override {
        return {
            ParamSpec::choice(QStringLiteral("method"), QStringLiteral("Background model"),
                              {QStringLiteral("Mean"), QStringLiteral("Median"), QStringLiteral("Σ-Δ"), QStringLiteral("Gaussian")}, 2)
                .withHelp(QStringLiteral("Mean: exponential average. Median: ±1 per frame toward the image. Σ-Δ: "
                                         "median plus a per-pixel spread V learnt the same way. Gaussian: running "
                                         "mean and variance.")),
            ParamSpec::real(QStringLiteral("alpha"), QStringLiteral("Learning rate α (Mean, Gaussian)"), 0.005, 0.2, 0.02, 0.005, 3)
                .withHelp(QStringLiteral("Share of the new frame blended into the estimate. High: adapts fast, but "
                                         "slow objects melt into the background and leave trails.")),
            ParamSpec::integer(QStringLiteral("tau"), QStringLiteral("Threshold τ (Mean, Median)"), 4, 80, 20, 1, QStringLiteral(" grey levels"))
                .withHelp(QStringLiteral("One fixed threshold for every pixel: too low flags noise, too high misses "
                                         "objects close to the background colour.")),
            ParamSpec::integer(QStringLiteral("n"), QStringLiteral("Amplification N (Σ-Δ, Gaussian)"), 1, 8, 2)
                .withHelp(QStringLiteral("Σ-Δ: V follows N·|I − B|. Gaussian: threshold at N·σ. Larger: fewer false "
                                         "alarms, more missed pixels.")),
            ParamSpec::choice(QStringLiteral("reg"), QStringLiteral("Regularisation"),
                              {QStringLiteral("None"), QStringLiteral("Morphology"), QStringLiteral("MRF")}, 2)
                .withHelp(QStringLiteral("Morphology: opening then closing with discs of radius 1 and 2. MRF: Potts "
                                         "prior on 8 neighbours, minimised by ICM from the raw mask.")),
            ParamSpec::real(QStringLiteral("beta"), QStringLiteral("MRF β (cost of a disagreeing neighbour)"), 0.0, 3.0, 1.0, 0.1, 1)
                .withHelp(QStringLiteral("0 gives back the raw test |I − B| > T; larger values erase bigger specks and "
                                         "round off thin parts.")),
            ParamSpec::toggle(QStringLiteral("bootstrap"), QStringLiteral("Start from a temporal median"), true)
                .withHelp(QStringLiteral("Initialise the model with the median of 15 frames spread over the clip. Off: "
                                         "start from frame 0, and whatever moves there leaves a ghost.")),
            ParamSpec::choice(QStringLiteral("baseline"), QStringLiteral("OpenCV baseline (Compare)"),
                              {QStringLiteral("MOG2"), QStringLiteral("KNN")}, 0)
                .withHelp(QStringLiteral("What Compare runs on the same frames, from the same start: a mixture of "
                                         "Gaussians per pixel (MOG2) or nearest neighbours among stored samples (KNN).")),
        };
    }

    std::vector<Sample> samples() const override { return motion_samples::background(); }

    Output compute(const Inputs& in, const Params& p, const RunContext& ctx) const override {
        const auto& frames = in.images;
        CV_Assert(!frames.empty());
        BgParams bp;
        bp.method = static_cast<BgMethod>(intParam(p, "method"));
        bp.alpha = float(realParam(p, "alpha"));
        bp.tau = float(intParam(p, "tau"));
        bp.amplification = intParam(p, "n");
        const auto reg = static_cast<Reg>(intParam(p, "reg"));
        MrfParams mp;
        mp.beta = realParam(p, "beta");
        const bool hasTruth = in.truth.size() == frames.size();
        const bool showThreshold = bp.method == BgMethod::SigmaDelta || bp.method == BgMethod::Gaussian;

        report(ctx.progress, 0.0, "initialising the background");
        const cv::Mat first = vl::toBGR(frames.front());
        const bool bootstrap = toggleParam(p, "bootstrap");
        const cv::Mat bg0 = bootstrap ? medianBackground(frames, 15) : first;
        // Noise from two consecutive frames: |I₁ − I₀| has √2 times the per-frame spread.
        const double noise = frames.size() > 1 ? noiseSigma(vl::toBGR(frames[1]), first) / std::sqrt(2.0) : 2.0;
        BackgroundModel model(bp, bg0, noise);
        std::optional<opencv::BackgroundSubtractor> baseline;
        const auto which = static_cast<opencv::Subtractor>(intParam(p, "baseline"));
        if (ctx.compare) baseline.emplace(which, bootstrap ? bg0 : cv::Mat());

        Output out;
        Scores ours, theirs;
        std::vector<double> changes;
        std::vector<double> lastEnergy;
        const QString thresholdName = bp.method == BgMethod::SigmaDelta ? QStringLiteral("V") : QStringLiteral("N·σ");

        for (size_t t = 0; t < frames.size(); ++t) {
            ctx.check();
            const cv::Mat frame = vl::toBGR(frames[t]);
            const cv::Mat* truth = hasTruth ? &in.truth[t] : nullptr;
            const Stopwatch clock;
            const BgStep st = model.apply(frame);
            cv::Mat fg = st.foreground;
            long changed = 0;
            if (reg == Reg::Morphology) {
                fg = alternatingSequentialFilter(st.foreground, 2);
                changed = cv::countNonZero(fg != st.foreground);
            } else if (reg == Reg::Mrf) {
                MrfResult r = icmPotts(detectionMargin(st.difference, st.threshold), st.foreground, mp, ctx.stop);
                fg = r.mask;
                changed = r.changes;
                lastEnergy = std::move(r.energy);
            }
            ours.ms += clock.ms();
            changes.push_back(double(changed));
            ours.add(fg, truth);

            Frame f;
            f.views.push_back({QStringLiteral("Frame"), toQImage(frame)});
            f.views.push_back({QStringLiteral("Background"), toQImage(model.background())});
            // Values under the cursor, kept as bytes to stay light over hundreds of frames.
            cv::Mat d8, t8;
            st.difference.convertTo(d8, CV_8U);
            st.threshold.convertTo(t8, CV_8U);
            const cv::Mat raw = st.foreground, fin = fg;
            const auto probe = [d8, t8, raw, fin](QPoint q) {
                const int dv = d8.at<uchar>(q.y(), q.x()), tv = t8.at<uchar>(q.y(), q.x());
                const bool r = raw.at<uchar>(q.y(), q.x()), k = fin.at<uchar>(q.y(), q.x());
                QString s = QStringLiteral("|I − B| = %1, T = %2 → %3").arg(dv).arg(tv).arg(r ? QStringLiteral("foreground") : QStringLiteral("background"));
                if (r != k) s += k ? QStringLiteral(" · filled in by the regularisation") : QStringLiteral(" · removed by the regularisation");
                return s;
            };
            for (View& v : f.views) v.probe = probe;
            f.views.push_back({QStringLiteral("Difference"), toQImage(vl::colorize(st.difference, 0, kDiffMax)), {}, probe, {},
                               Legend{QStringLiteral("|I − B| (grey levels, mean of B, G, R)"), 0, kDiffMax, {}, {}}});
            if (showThreshold) {
                f.views.push_back({QStringLiteral("Threshold"), toQImage(vl::colorize(st.threshold, 0, kDiffMax)), {}, probe, {},
                                   Legend{QStringLiteral("threshold T = %1 (grey levels)").arg(thresholdName), 0, kDiffMax, {}, {}}});
            }

            // Result: kept pixels in the accent colour; what the regularisation removed in pink, added in blue.
            cv::Mat pic = dimmed(frame);
            pic.setTo(bgr(colors::accent), fg);
            pic.setTo(bgr(colors::bad, 0.6), raw & ~fg);  // dimmer: context, not the result
            pic.setTo(bgr(colors::info), fg & ~raw);
            View result{QStringLiteral("Foreground"), toQImage(pic), {}, probe};
            if (truth) {
                result.shapes = outline(*truth, colors::good);
                f.stats.push_back({QStringLiteral("F1 (this frame)"), fmt(ours.f1.back(), 3), false});
                f.stats.push_back({QStringLiteral("Precision / recall"), fmt(ours.precision.back(), 2) + QStringLiteral(" / ") + fmt(ours.recall.back(), 2)});
            }
            f.views.push_back(result);

            if (baseline) {
                const auto b = baseline->apply(frame);
                theirs.ms += b.ms;
                theirs.add(b.foreground, truth);
                cv::Mat cvPic = dimmed(frame);
                cvPic.setTo(bgr(colors::accent), b.foreground);
                View v{QStringLiteral("OpenCV · Foreground"), toQImage(cvPic)};
                if (truth) {
                    v.shapes = outline(*truth, colors::good);
                    f.stats.push_back({QStringLiteral("OpenCV F1 (this frame)"), fmt(theirs.f1.back(), 3)});
                }
                const cv::Mat mask = b.foreground;
                v.probe = [mask](QPoint q) {
                    return mask.at<uchar>(q.y(), q.x()) ? QStringLiteral("OpenCV: foreground") : QStringLiteral("OpenCV: background");
                };
                f.views.push_back(std::move(v));
            }
            f.stats.push_back({QStringLiteral("Foreground share"), fmt(100 * ours.share.back()) + QStringLiteral(" %")});
            if (reg != Reg::None) f.stats.push_back({QStringLiteral("Regularisation changes"), QStringLiteral("%1 px").arg(changed)});
            out.frames.push_back(std::move(f));

            report(ctx.progress, double(t + 1) / double(frames.size()), "frame " + std::to_string(t + 1) + " / " + std::to_string(frames.size()));
            if (ctx.partial && motion_samples::partialDue(t)) {
                auto part = std::make_shared<Output>(out);
                summarise(*part, ours, changes, lastEnergy, reg, hasTruth);
                ctx.partial(part);
            }
        }
        summarise(out, ours, changes, lastEnergy, reg, hasTruth);
        if (baseline) compareWithOpenCv(out, ours, theirs, hasTruth);
        return out;
    }

private:
    static void summarise(Output& out, const Scores& s, const std::vector<double>& changes,
                          const std::vector<double>& energy, Reg reg, bool hasTruth) {
        out.stats.clear();
        out.series.clear();
        const double share = mean(s.share);
        if (hasTruth) {
            out.stats.push_back({QStringLiteral("Mean F1 over the clip"), fmt(mean(s.f1), 3), true});
            out.series.push_back({QStringLiteral("F1 per frame"), s.f1});
        }
        out.stats.push_back({QStringLiteral("Mean foreground share"), fmt(100 * share) + QStringLiteral(" %"), !hasTruth});
        if (reg != Reg::None)
            out.stats.push_back({QStringLiteral("Regularisation changes"), fmt(mean(changes), 0) + QStringLiteral(" px/frame")});
        out.stats.push_back({QStringLiteral("Time per frame"), fmtMs(s.ms / double(std::max<size_t>(1, s.share.size())))});
        out.series.push_back({QStringLiteral("Foreground share per frame"), s.share});
        if (reg == Reg::Mrf && energy.size() > 1) out.series.push_back({QStringLiteral("MRF energy per ICM sweep (last frame)"), energy});
        out.note.clear();
        if (share > 0.25)
            out.note = QStringLiteral("A quarter of the picture is flagged on average: the camera probably moves, or "
                                      "the threshold is too low.");
    }

    // The same measurements for ours and OpenCV's subtractor, side by side.
    static void compareWithOpenCv(Output& out, const Scores& ours, const Scores& theirs, bool hasTruth) {
        const double n = double(std::max<size_t>(1, ours.share.size()));
        if (hasTruth) {
            out.comparison.push_back({QStringLiteral("Mean F1"), mean(ours.f1), mean(theirs.f1), {}, 3, false});
            out.comparison.push_back({QStringLiteral("Precision"), mean(ours.precision), mean(theirs.precision), {}, 3, false});
            out.comparison.push_back({QStringLiteral("Recall"), mean(ours.recall), mean(theirs.recall), {}, 3, false});
        }
        out.comparison.push_back({QStringLiteral("Foreground share"), 100 * mean(ours.share), 100 * mean(theirs.share),
                                  QStringLiteral(" %"), 1, true});
        out.comparison.push_back({QStringLiteral("Time per frame"), ours.ms / n, theirs.ms / n, QStringLiteral(" ms"), 2});
    }
};

}  // namespace

VL_REGISTER_MODULE(BackgroundModule);

}  // namespace vl::app
