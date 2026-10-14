// Topic: Horn–Schunck, a global variational flow: one energy, data term + smoothness, minimised by
// iterations whose progress is shown live.

#include <memory>

#include "app/module.hpp"
#include "app/modules/flow/flow_samples.hpp"
#include "app/modules/flow/flow_views.hpp"
#include "core/common/image.hpp"
#include "core/flow/flow_eval.hpp"
#include "core/flow/horn_schunck.hpp"
#include "core/flow/opencv_baselines.hpp"

namespace vl::app {
namespace {

using namespace flow_views;

constexpr double kPartialMs = 200;  // computing time between two partial pictures

class HornSchunckModule final : public Module {
public:
    Info info() const override {
        return {QStringLiteral("flow-hs"), QStringLiteral("Horn–Schunck"), QStringLiteral("Optical flow"),
                QStringLiteral("Instead of solving each window alone, look for the whole field that best explains the "
                               "frames <i>and</i> varies smoothly: E = Σ(I<sub>x</sub>u + I<sub>y</sub>v + "
                               "I<sub>t</sub>)² + α² Σ(|∇u|² + |∇v|²). Each iteration pulls a pixel towards the mean "
                               "of its neighbours, corrected by its own brightness constraint, so motion "
                               "<b>spreads into</b> flat areas, and across motion boundaries too."),
                QStringLiteral("on <i>Synthetic: flat square</i>, compare Flow with Lucas–Kanade's: the untextured "
                               "inside gets filled in. Then raise α to 40 and watch the square's outline blur."),
                InputKind::ImagePair, 720, QStringLiteral("Flow"), QStringLiteral("cv::calcOpticalFlowFarneback"),
                {QStringLiteral("src/core/flow/horn_schunck.hpp"), QStringLiteral("src/core/flow/horn_schunck.cpp"),
                 QStringLiteral("src/core/flow/flow_common.hpp"), QStringLiteral("src/core/flow/flow_common.cpp"),
                 QStringLiteral("src/core/common/filters.hpp"), QStringLiteral("src/core/common/filters.cpp")}};
    }

    std::vector<ParamSpec> params() const override {
        return {
            ParamSpec::real(QStringLiteral("alpha"), QStringLiteral("Smoothness α"), 1, 60, 12, 1, 0)
                .withHelp(QStringLiteral("Weight of smoothness against the data, in grey levels per pixel of flow "
                                         "change. Small: noisy, holes in flat areas. Large: smooth, blurred boundaries.")),
            ParamSpec::integer(QStringLiteral("iterations"), QStringLiteral("Sweeps per level"), 10, 500, 150, 10)
                .withHelp(QStringLiteral("Each sweep updates every pixel once. Information travels about one pixel "
                                         "per sweep, so filling a large flat area takes many.")),
            ParamSpec::choice(QStringLiteral("solver"), QStringLiteral("Update"), {"Jacobi", "Gauss–Seidel"}, 1)
                .withHelp(QStringLiteral("Jacobi updates all pixels from the previous field (Horn and Schunck's "
                                         "original); Gauss–Seidel uses new values at once: never raises the energy, "
                                         "about twice as fast.")),
            ParamSpec::toggle(QStringLiteral("c2f"), QStringLiteral("Coarse to fine"), true)
                .withHelp(QStringLiteral("Solve on halved copies first and re-warp at each level. Off: one scale, "
                                         "which only handles motions of a pixel or two.")),
            ParamSpec::integer(QStringLiteral("levels"), QStringLiteral("Pyramid levels"), 2, 6, 4),
            ParamSpec::integer(QStringLiteral("arrows"), QStringLiteral("Arrow spacing"), 8, 48, 16, 4, QStringLiteral(" px")),
        };
    }

    std::vector<Sample> samples() const override { return flow_samples::all(); }

    Output compute(const Inputs& in, const Params& p, const RunContext& ctx) const override {
        vl::flow::HornSchunckParams hp;
        hp.alpha = realParam(p, "alpha");
        hp.iterations = intParam(p, "iterations");
        hp.solver = static_cast<vl::flow::HornSchunckSolver>(intParam(p, "solver"));
        const bool c2f = toggleParam(p, "c2f");
        hp.levels = c2f ? intParam(p, "levels") : 1;
        hp.reportEvery = 25;  // each report brings the field to full resolution: not too often

        const Presenter pres(in, intParam(p, "arrows"));
        const auto placeholder = [&] {
            View v{QStringLiteral("Data weight"), toQImage(in.images[0] * 0.35)};  // filled in at the end
            return v;
        };

        // Partial pictures while it converges: the flow so far, at full resolution.
        Stopwatch since;
        bool first = true;
        double picturesMs = 0;  // time spent drawing partial results, left out of the run time
        const auto onStep = [&](const cv::Mat& flow, const std::string& stage) {
            if (!ctx.partial || (!first && since.ms() < kPartialMs)) return;
            const Stopwatch drawing;
            first = false;
            auto out = std::make_shared<Output>(pres.present(flow, placeholder()));
            out->note = QStringLiteral("Converging: %1").arg(QString::fromStdString(stage));
            ctx.partial(out);
            picturesMs += drawing.ms();
            since = Stopwatch();  // count from here, so building pictures never dominates the run
        };
        const Stopwatch clock;
        const auto res = vl::flow::hornSchunck(in.images[0], in.images[1], hp, ctx.stop, ctx.progress, onStep);
        const double oursMs = clock.ms() - picturesMs;

        // Mechanism: how much of each pixel's update comes from its own data rather than its neighbours.
        const cv::Mat dw = res.dataWeight;
        View mech{QStringLiteral("Data weight"), toQImage(vl::colorize(dw, 0, 1)), {}, {}, dw,
                  Legend{QStringLiteral("|∇I|² / (|∇I|² + α²): share of the update taken from the pixel's own data"), 0, 1,
                         QStringLiteral("0: copied from neighbours"), QStringLiteral("1: data")}};
        mech.probe = [dw](QPoint q) {
            if (q.x() < 0 || q.y() < 0 || q.x() >= dw.cols || q.y() >= dw.rows) return QString();
            const double v = dw.at<float>(q.y(), q.x());
            return QStringLiteral("data weight %1  ·  %2").arg(v, 0, 'f', 2).arg(
                v < 0.2 ? QStringLiteral("flow mostly filled in from the neighbours") : QStringLiteral("flow mostly measured here"));
        };

        Output out = pres.present(res.flow, mech);
        out.stats.push_back({QStringLiteral("Sweeps (all levels)"), QString::number(res.sweeps)});
        out.series.push_back({QStringLiteral("Energy per pixel, finest level"), res.energy});
        out.series.push_back({QStringLiteral("Mean update per sweep (px)"), res.meanUpdate});
        if (!c2f && pres.truthMax() > 2.5)
            out.note = QStringLiteral("The true motion reaches %1 px; one scale only linearises about a pixel or two. "
                                      "Turn coarse to fine on.").arg(pres.truthMax(), 0, 'f', 1);
        if (ctx.compare) {
            // Farnebäck, OpenCV's classic dense method, with the same number of pyramid levels.
            const auto baseline = vl::flow::opencv::farneback(in.images[0], in.images[1], hp.levels);
            addComparison(out, in, pres, res.flow, oursMs, baseline.flow, baseline.ms, QStringLiteral("calcOpticalFlowFarneback"),
                          grossErrors(in, pres, res.flow, baseline.flow));
        }
        return out;
    }

    // Share of pixels off by more than 3 px: where smoothing across a motion boundary costs most.
    static ExtraComparison grossErrors(const Inputs& in, const Presenter& pres, const cv::Mat& ours, const cv::Mat& theirs) {
        if (!pres.hasTruth()) return {};
        return Comparison{QStringLiteral("Pixels with error > 3 px"), 100 * vl::flow::evaluateFlow(ours, in.truth[0]).over3,
                          100 * vl::flow::evaluateFlow(theirs, in.truth[0]).over3, QStringLiteral(" %"), 1};
    }

    std::vector<ClickShapes> onClick(const Output& out, int, int, QPointF px) const override {
        return clickShapes(out, px);
    }
};

}  // namespace

VL_REGISTER_MODULE(HornSchunckModule);

}  // namespace vl::app
