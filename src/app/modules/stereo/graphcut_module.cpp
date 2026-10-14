// Topic: global stereo with occlusions by graph cuts (Kolmogorov–Zabih, α-expansion on assignments).

#include <cmath>

#include "app/module.hpp"
#include "app/modules/stereo/stereo_samples.hpp"
#include "app/modules/stereo/stereo_views.hpp"
#include "core/stereo/block_matching.hpp"
#include "core/stereo/graphcut_stereo.hpp"

namespace vl::app {
namespace {

namespace sv = stereo_views;

enum ViewIndex { kLeft, kRight, kDisparity, kOcclusions, kErrors };

class GraphCutModule final : public Module {
public:
    Info info() const override {
        return {QStringLiteral("stereo-graphcut"), QStringLiteral("Graph cuts"), QStringLiteral("Stereo"),
                QStringLiteral("Instead of choosing each pixel's disparity alone, minimise one energy over the whole "
                               "map: matched pixels should look alike, neighbours should share a disparity, and every "
                               "pixel left unmatched in <i>either</i> image pays a fixed occlusion cost. Each step "
                               "(an <b>α-expansion</b>) lets every pixel switch to disparity α or drop its match, and a "
                               "single minimum cut finds the best such step exactly."),
                QStringLiteral("on <i>Tsukuba</i>, open <i>Occlusions</i> and turn <i>Model occlusions</i> off: the "
                               "blue strips beside each object are hidden in the right image, yet now get a depth. Then "
                               "set λ to 0.5 and see the noise come back."),
                InputKind::ImagePair, 320, QStringLiteral("Disparity")};
    }

    std::vector<ParamSpec> params() const override {
        return {
            ParamSpec::real(QStringLiteral("lambda"), QStringLiteral("Smoothness λ"), 0.5, 10, 2, 0.5, 1)
                .withHelp(QStringLiteral("Penalty when two neighbouring pixels take different disparities, in grey "
                                         "levels of matching cost. Higher: flatter, blockier maps.")),
            ParamSpec::real(QStringLiteral("occlusion"), QStringLiteral("Occlusion cost K"), 1, 30, 6, 1, 0)
                .withHelp(QStringLiteral("Price of leaving a pixel unmatched. Too low and textureless areas give up; "
                                         "too high and occluded strips get forced onto wrong matches.")),
            ParamSpec::toggle(QStringLiteral("occlusions"), QStringLiteral("Model occlusions"), true)
                .withHelp(QStringLiteral("Off: the classic Potts labelling (Boykov–Veksler–Zabih), one disparity for "
                                         "every left pixel, no uniqueness, no occlusion term.")),
            ParamSpec::choice(QStringLiteral("cost"), QStringLiteral("Data cost"), {QStringLiteral("|ΔI|"), QStringLiteral("ΔI²")}, 0)
                .withHelp(QStringLiteral("Difference of colours, truncated at 30 grey levels; the squared cost is "
                                         "rescaled to the same range, so it forgives small differences more.")),
            ParamSpec::toggle(QStringLiteral("bt"), QStringLiteral("Birchfield–Tomasi cost"), true)
                .withHelp(QStringLiteral("Compare each pixel with the interpolated signal within half a pixel, so that "
                                         "a match that falls between two samples is not penalised.")),
            ParamSpec::toggle(QStringLiteral("adaptive"), QStringLiteral("Adaptive λ at image edges"), true)
                .withHelp(QStringLiteral("Divide λ by 3 between neighbours whose intensities differ by 8 or more, "
                                         "in either image, where a depth edge is likely.")),
            ParamSpec::integer(QStringLiteral("iterations"), QStringLiteral("Passes over disparities"), 1, 6, 3)
                .withHelp(QStringLiteral("Each pass tries one expansion per disparity; it stops early when a pass "
                                         "changes nothing.")),
            ParamSpec::choice(QStringLiteral("scale"), QStringLiteral("Resolution"), {QStringLiteral("Full"), QStringLiteral("½"), QStringLiteral("¼")}, 1)
                .withHelp(QStringLiteral("Work on a smaller copy of the pair: about 8 times faster per halving. "
                                         "Errors are measured at that resolution.")),
        };
    }

    std::vector<Sample> samples() const override { return stereo_samples::all(); }

    Output compute(const Inputs& in, const Params& p, const RunContext& ctx) const override {
        const int factor = 1 << intParam(p, "scale");
        const sv::Working w = sv::prepare(in, factor);

        stereo::GraphCutParams gp;
        gp.maxDisparity = w.maxDisparity;
        gp.lambda = realParam(p, "lambda");
        gp.occlusionCost = realParam(p, "occlusion");
        gp.modelOcclusions = toggleParam(p, "occlusions");
        gp.cost = static_cast<stereo::DataCost>(intParam(p, "cost"));
        gp.birchfieldTomasi = toggleParam(p, "bt");
        gp.adaptiveLambda = toggleParam(p, "adaptive");
        gp.maxIterations = intParam(p, "iterations");

        // Partial pictures while the expansions run, at most every 120 ms.
        vl::Stopwatch clock;
        double lastPartial = -1e9;
        std::vector<double> energies;
        auto observer = [&](const stereo::GraphCutStep& st) {
            energies.push_back(st.energy);
            if (!ctx.partial || clock.ms() - lastPartial < 120) return;
            lastPartial = clock.ms();
            auto out = std::make_shared<Output>();
            out->frames.push_back(frame(w, *st.disparity));
            out->stats.push_back({QStringLiteral("Pass %1, expanding d = %2").arg(st.iteration).arg(st.alpha),
                                  QStringLiteral("energy %1").arg(fmt(st.energy, 0)), true});
            out->series.push_back({QStringLiteral("Energy after each expansion move"), energies});
            ctx.partial(out);
        };
        const auto res = stereo::graphCutStereo(w.left, w.right, gp, ctx.stop,
                                                [&](double f, const std::string& m) { if (ctx.progress) ctx.progress(0.95 * f, m); },
                                                observer);

        Output out;
        out.frames.push_back(frame(w, res.disparity));
        if (!w.truth.empty()) {
            out.stats.push_back({QStringLiteral("Bad pixels (non-occluded)"),
                                 fmt(100 * sv::badRate(res.disparity, w)) + QStringLiteral(" %"), true});
            // The local method on the same pair, as the "before" number.
            stereo::BlockMatchParams bp;
            bp.maxDisparity = w.maxDisparity;
            bp.radius = 3;
            const auto bm = stereo::blockMatch(w.left, w.right, bp, ctx.stop);
            out.stats.push_back({QStringLiteral("Block matching, same pair"),
                                 fmt(100 * sv::badRate(bm.disparity, w)) + QStringLiteral(" %")});
        }
        out.stats.push_back({QStringLiteral("Energy start → end"),
                             fmt(res.energy.front(), 0) + QStringLiteral(" → ") + fmt(res.energy.back(), 0)});
        out.stats.push_back({QStringLiteral("Expansion moves"),
                             QStringLiteral("%1 · %2 pass%3%4")
                                 .arg(res.moves)
                                 .arg(res.iterations)
                                 .arg(res.iterations > 1 ? QStringLiteral("es") : QString())
                                 .arg(res.converged ? QStringLiteral(" · converged") : QString())});
        if (gp.modelOcclusions)
            out.stats.push_back({QStringLiteral("Occluded, left · right"),
                                 fmt(100.0 * cv::countNonZero(res.occluded) / double(res.occluded.total())) +
                                     QStringLiteral(" % · ") +
                                     fmt(100.0 * cv::countNonZero(res.occludedRight) / double(res.occludedRight.total())) +
                                     QStringLiteral(" %")});
        out.series.push_back({QStringLiteral("Energy after each expansion move"),
                              std::vector<double>(res.energy.begin() + 1, res.energy.end())});
        if (factor > 1)
            out.note = QStringLiteral("Computed at %1 resolution (%2 × %3 px); choose Full for the finest map, about %4× "
                                      "slower.")
                           .arg(factor == 2 ? QStringLiteral("half") : QStringLiteral("quarter"))
                           .arg(w.left.cols)
                           .arg(w.left.rows)
                           .arg(factor == 2 ? 8 : 60);
        return out;
    }

    std::vector<ClickShapes> onClick(const Output& out, int, int view, QPointF px) const override {
        if (out.frames.empty() || view == kRight) return {};
        const auto& f = out.frames.front();
        return sv::matchClick(f.views[kDisparity].data, kRight, int(f.views.size()), px,
                              QStringLiteral("occluded: no match in this image"));
    }

private:
    static Frame frame(const sv::Working& w, const cv::Mat& disparity) {
        Frame f;
        View left{QStringLiteral("Left"), toQImage(w.left)};
        View right{QStringLiteral("Right"), toQImage(w.right)};
        View disp = sv::disparityView(QStringLiteral("Disparity"), disparity, w, QStringLiteral("occluded"));
        left.probe = disp.probe;
        f.views = {left, right, disp, occlusionView(w, disparity)};
        if (!w.truth.empty()) f.views.push_back(View{QStringLiteral("Errors"), toQImage(sv::errorImage(disparity, w)), {}, disp.probe, disparity});
        return f;
    }

    // Occluded pixels over the dimmed left image. With ground truth: green where the occlusion is real,
    // pink where the pixel is in fact visible, blue for real occlusions that received a match anyway.
    static View occlusionView(const sv::Working& w, const cv::Mat& disparity) {
        cv::Mat img = sv::dimmed(w.left);
        const bool truth = !w.truth.empty();
        const cv::Vec3b good{0x7b, 0xd9, 0x4c}, bad{0x8a, 0x5c, 0xff}, info{0xf7, 0xc3, 0x4f}, accent{0x00, 0xb0, 0xff};
        int hit = 0, found = 0, real = 0;
        for (int y = 0; y < img.rows; ++y)
            for (int x = 0; x < img.cols; ++x) {
                const bool occ = !std::isfinite(disparity.at<float>(y, x));
                const bool isReal = truth && w.truth.at<float>(y, x) > 0 && w.visible.at<uchar>(y, x) == 0;
                found += occ;
                real += isReal;
                hit += occ && isReal;
                cv::Vec3b& px = img.at<cv::Vec3b>(y, x);
                if (!truth) {
                    if (occ) px = accent;
                } else if (occ) {
                    px = isReal ? good : bad;
                } else if (isReal) {
                    px = info;
                }
            }
        View v{QStringLiteral("Occlusions"), toQImage(img)};
        const cv::Mat d = disparity, vis = w.visible, t = w.truth;
        v.probe = [d, vis, t](QPoint q) {
            if (q.x() < 0 || q.y() < 0 || q.x() >= d.cols || q.y() >= d.rows) return QString();
            const bool occ = !std::isfinite(d.at<float>(q.y(), q.x()));
            if (vis.empty()) return occ ? QStringLiteral("occluded") : QStringLiteral("matched");
            const bool hidden = t.at<float>(q.y(), q.x()) > 0 && vis.at<uchar>(q.y(), q.x()) == 0;
            if (occ) return hidden ? QStringLiteral("occluded, and truly hidden in the right image")
                                 : QStringLiteral("occluded, but visible in the right image");
            return hidden ? QStringLiteral("truly hidden in the right image, yet matched")
                        : QStringLiteral("matched, d = %1 px").arg(d.at<float>(q.y(), q.x()), 0, 'f', 0);
        };
        if (truth && real > 0 && found == 0)
            v.shapes.push_back(LabelShape{QPointF(6, 6), QStringLiteral("no occlusions: all %1 hidden pixels (blue) received a match")
                                                             .arg(real)});
        else if (truth && real > 0)
            v.shapes.push_back(LabelShape{QPointF(6, 6),
                                          QStringLiteral("found %1 % of the true occlusions · %2 % of the found ones are real")
                                              .arg(100.0 * hit / real, 0, 'f', 0)
                                              .arg(found ? 100.0 * hit / found : 0.0, 0, 'f', 0)});
        return v;
    }
};

}  // namespace

VL_REGISTER_MODULE(GraphCutModule);

}  // namespace vl::app
