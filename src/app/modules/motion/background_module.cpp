// Topic: background subtraction with a static camera (mean, approximate median, Σ-Δ, Gaussian),
// with morphological or Markovian regularisation of the foreground mask.

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

#include "app/module.hpp"
#include "app/modules/motion/motion_samples.hpp"
#include "core/common/image.hpp"
#include "core/motion/background.hpp"
#include "core/motion/regularize.hpp"

namespace vl::app {
namespace {

using namespace vl::motion;

constexpr double kDiffMax = 48.0;  // top of the colour scale for |I − M| and T, grey levels

enum class Reg { None, Morphology, Mrf };

// Dimmed grey copy of the frame, so coloured masks read clearly on top of it.
cv::Mat dimmed(const cv::Mat& bgr) {
    cv::Mat g, out;
    cv::cvtColor(bgr, g, cv::COLOR_BGR2GRAY);
    cv::cvtColor(g * 0.45, out, cv::COLOR_GRAY2BGR);
    return out;
}

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
                InputKind::Video, 500, QStringLiteral("Foreground")};
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
                .withHelp(QStringLiteral("Σ-Δ: V follows N·|I − M|. Gaussian: threshold at N·σ. Larger: fewer false "
                                         "alarms, more missed pixels.")),
            ParamSpec::choice(QStringLiteral("reg"), QStringLiteral("Regularisation"),
                              {QStringLiteral("None"), QStringLiteral("Morphology"), QStringLiteral("MRF")}, 2)
                .withHelp(QStringLiteral("Morphology: opening then closing with discs of radius 1 and 2. MRF: Potts "
                                         "prior on 8 neighbours, minimised by ICM from the raw mask.")),
            ParamSpec::real(QStringLiteral("beta"), QStringLiteral("MRF β (cost of a disagreeing neighbour)"), 0.0, 3.0, 1.0, 0.1, 1)
                .withHelp(QStringLiteral("0 gives back the raw test |I − M| > T; larger values erase bigger specks and "
                                         "round off thin parts.")),
            ParamSpec::toggle(QStringLiteral("bootstrap"), QStringLiteral("Start from a temporal median"), true)
                .withHelp(QStringLiteral("Initialise the model with the median of 15 frames spread over the clip. Off: "
                                         "start from frame 0, and whatever moves there leaves a ghost.")),
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
        const cv::Mat bg0 = toggleParam(p, "bootstrap") ? medianBackground(frames, 15) : first;
        // Noise from two consecutive frames: |I₁ − I₀| has √2 times the per-frame spread.
        const double noise = frames.size() > 1 ? noiseSigma(vl::toBGR(frames[1]), first) / std::sqrt(2.0) : 2.0;
        BackgroundModel model(bp, bg0, noise);

        Output out;
        std::vector<double> f1s, shares, changes;
        std::vector<double> lastEnergy;
        Stopwatch sw;
        const QString thresholdName = bp.method == BgMethod::SigmaDelta ? QStringLiteral("V") : QStringLiteral("N·σ");

        for (size_t t = 0; t < frames.size(); ++t) {
            ctx.check();
            const cv::Mat frame = vl::toBGR(frames[t]);
            const BgStep st = model.apply(frame);
            cv::Mat fg = st.foreground;
            long changed = 0;
            if (reg == Reg::Morphology) {
                fg = alternatingSequentialFilter(st.foreground, 2);
                cv::Mat d;
                cv::compare(fg, st.foreground, d, cv::CMP_NE);
                changed = cv::countNonZero(d);
            } else if (reg == Reg::Mrf) {
                MrfResult r = icmPotts(detectionMargin(st.difference, st.threshold), st.foreground, mp, ctx.stop);
                fg = r.mask;
                changed = r.changes;
                lastEnergy = std::move(r.energy);
            }
            changes.push_back(double(changed));
            const double share = cv::countNonZero(fg) / double(fg.total());
            shares.push_back(share);

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
                QString s = QStringLiteral("|I − M| = %1, T = %2 → %3").arg(dv).arg(tv).arg(r ? QStringLiteral("foreground") : QStringLiteral("background"));
                if (r != k) s += k ? QStringLiteral(" · filled in by the regularisation") : QStringLiteral(" · removed by the regularisation");
                return s;
            };
            for (View& v : f.views) v.probe = probe;
            View diff{QStringLiteral("Difference"), toQImage(vl::colorize(st.difference, 0, kDiffMax)), {}, probe, {},
                      Legend{QStringLiteral("|I − M| (grey levels, mean of B, G, R)"), 0, kDiffMax, {}, {}}};
            f.views.push_back(diff);
            if (showThreshold) {
                f.views.push_back({QStringLiteral("Threshold"), toQImage(vl::colorize(st.threshold, 0, kDiffMax)), {}, probe, {},
                                   Legend{QStringLiteral("threshold T = %1 (grey levels)").arg(thresholdName), 0, kDiffMax, {}, {}}});
            }

            // Result: kept pixels in the accent colour; what the regularisation removed in pink, added in blue.
            cv::Mat pic = dimmed(frame);
            const auto bgr = [](const QColor& c, double k) { return cv::Scalar(k * c.blue(), k * c.green(), k * c.red()); };
            pic.setTo(bgr(colors::accent, 1.0), fg);
            pic.setTo(bgr(colors::bad, 0.6), raw & ~fg);  // dimmer: context, not the result
            pic.setTo(bgr(colors::info, 1.0), fg & ~raw);
            View result{QStringLiteral("Foreground"), toQImage(pic), {}, probe};
            if (hasTruth) {
                result.shapes = outline(in.truth[t], colors::good);
                const Confusion c = compareMasks(fg, in.truth[t]);
                f1s.push_back(c.f1());
                f.stats.push_back({QStringLiteral("F1 (this frame)"), fmt(c.f1(), 3), false});
                f.stats.push_back({QStringLiteral("Precision / recall"), fmt(c.precision(), 2) + QStringLiteral(" / ") + fmt(c.recall(), 2)});
            }
            f.views.push_back(result);
            f.stats.push_back({QStringLiteral("Foreground share"), fmt(100 * share) + QStringLiteral(" %")});
            if (reg != Reg::None) f.stats.push_back({QStringLiteral("Regularisation changes"), QStringLiteral("%1 px").arg(changed)});
            out.frames.push_back(std::move(f));

            report(ctx.progress, double(t + 1) / double(frames.size()), "frame " + std::to_string(t + 1) + " / " + std::to_string(frames.size()));
            if (ctx.partial && motion_samples::partialDue(t)) {
                auto part = std::make_shared<Output>(out);
                summarise(*part, f1s, shares, changes, lastEnergy, reg, sw.ms(), hasTruth);
                ctx.partial(part);
            }
        }
        summarise(out, f1s, shares, changes, lastEnergy, reg, sw.ms(), hasTruth);
        return out;
    }

private:
    static double mean(const std::vector<double>& v) {
        double s = 0;
        for (double x : v) s += x;
        return v.empty() ? 0.0 : s / double(v.size());
    }

    static void summarise(Output& out, const std::vector<double>& f1s, const std::vector<double>& shares,
                          const std::vector<double>& changes, const std::vector<double>& energy, Reg reg, double ms,
                          bool hasTruth) {
        out.stats.clear();
        out.series.clear();
        const double share = mean(shares);
        if (hasTruth) {
            out.stats.push_back({QStringLiteral("Mean F1 over the clip"), fmt(mean(f1s), 3), true});
            out.series.push_back({QStringLiteral("F1 per frame"), f1s});
        }
        out.stats.push_back({QStringLiteral("Mean foreground share"), fmt(100 * share) + QStringLiteral(" %"), !hasTruth});
        if (reg != Reg::None)
            out.stats.push_back({QStringLiteral("Regularisation changes"), fmt(mean(changes), 0) + QStringLiteral(" px/frame")});
        out.stats.push_back({QStringLiteral("Time per frame"), fmtMs(ms / double(std::max<size_t>(1, shares.size())))});
        out.series.push_back({QStringLiteral("Foreground share per frame"), shares});
        if (reg == Reg::Mrf && energy.size() > 1) out.series.push_back({QStringLiteral("MRF energy per ICM sweep (last frame)"), energy});
        out.note.clear();
        if (share > 0.25)
            out.note = QStringLiteral("A quarter of the picture is flagged on average: the camera probably moves, or "
                                      "the threshold is too low.");
    }
};

}  // namespace

VL_REGISTER_MODULE(BackgroundModule);

}  // namespace vl::app
