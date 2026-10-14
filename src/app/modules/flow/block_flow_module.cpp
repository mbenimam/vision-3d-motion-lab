// Topic: optical flow by block matching, the brute-force baseline the variational methods improve on.

#include <algorithm>
#include <cmath>
#include <vector>

#include "app/module.hpp"
#include "app/modules/flow/flow_samples.hpp"
#include "app/modules/flow/flow_views.hpp"
#include "core/common/image.hpp"
#include "core/flow/block_flow.hpp"
#include "core/flow/flow_eval.hpp"
#include "core/flow/opencv_baselines.hpp"

namespace vl::app {
namespace {

using namespace flow_views;

class BlockFlowModule final : public Module {
public:
    Info info() const override {
        return {QStringLiteral("flow-block"), QStringLiteral("Block matching"), QStringLiteral("Optical flow"),
                QStringLiteral("For each pixel, take the small block around it in frame 1 and try every displacement "
                               "of a search window in frame 2; the flow is the displacement whose block looks most "
                               "alike. Exhaustive and honest, but blind where a block has no texture or only a "
                               "straight edge: many displacements fit equally well."),
                QStringLiteral("on <i>Synthetic: flat square</i>, click inside the square: the cost dots in Frame 2 are "
                               "flat, every displacement fits. Click its top edge: a valley along the edge."),
                InputKind::ImagePair, 700, QStringLiteral("Flow"), QStringLiteral("cv::DISOpticalFlow"),
                {QStringLiteral("src/core/flow/block_flow.hpp"), QStringLiteral("src/core/flow/block_flow.cpp"),
                 QStringLiteral("src/core/common/filters.hpp")}};
    }

    std::vector<ParamSpec> params() const override {
        return {
            ParamSpec::integer(QStringLiteral("radius"), QStringLiteral("Block radius"), 1, 10, 4, 1, QStringLiteral(" px"))
                .withHelp(QStringLiteral("The block is (2r+1)² pixels. Larger blocks see more texture but straddle "
                                         "motion boundaries.")),
            ParamSpec::integer(QStringLiteral("search"), QStringLiteral("Search radius"), 2, 24, 12, 1, QStringLiteral(" px"))
                .withHelp(QStringLiteral("Displacements tried in [−S, S]²: the cost grows with (2S+1)², and a motion "
                                         "beyond S cannot be found.")),
            ParamSpec::choice(QStringLiteral("cost"), QStringLiteral("Block difference"), {"SAD", "SSD"}, 0)
                .withHelp(QStringLiteral("Sum of absolute or of squared grey-level differences over the block.")),
            ParamSpec::toggle(QStringLiteral("subpixel"), QStringLiteral("Sub-pixel refinement"), true)
                .withHelp(QStringLiteral("Fit a parabola through the costs of the neighbouring displacements; without it "
                                         "the flow is quantised to whole pixels.")),
            ParamSpec::integer(QStringLiteral("arrows"), QStringLiteral("Arrow spacing"), 8, 48, 16, 4, QStringLiteral(" px")),
        };
    }

    std::vector<Sample> samples() const override { return flow_samples::all(); }

    Output compute(const Inputs& in, const Params& p, const RunContext& ctx) const override {
        vl::flow::BlockFlowParams bp;
        bp.radius = intParam(p, "radius");
        bp.search = intParam(p, "search");
        bp.cost = static_cast<vl::flow::BlockCost>(intParam(p, "cost"));
        bp.subpixel = toggleParam(p, "subpixel");
        const vl::Stopwatch clock;
        const auto res = vl::flow::blockMatchFlow(in.images[0], in.images[1], bp, ctx.stop, ctx.progress);
        const double oursMs = clock.ms();

        // Mechanism: how well the best block fits. High where nothing matches (occlusions, motion out of range).
        const cv::Mat cost = res.cost;
        std::vector<float> v(cost.begin<float>(), cost.end<float>());
        std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(v.size() * 95 / 100), v.end());
        const double hi = std::max(1.0, double(v[v.size() * 95 / 100]));
        const QString unit = bp.cost == vl::flow::BlockCost::SAD ? QStringLiteral("grey levels") : QStringLiteral("grey levels²");
        View mech{QStringLiteral("Match cost"), toQImage(vl::colorize(cost, 0, hi)), {}, {}, cost,
                  Legend{QStringLiteral("best block cost, mean per pixel (%1)").arg(unit), 0, hi, {}, {}}};
        mech.probe = [cost, unit](QPoint q) {
            if (q.x() < 0 || q.y() < 0 || q.x() >= cost.cols || q.y() >= cost.rows) return QString();
            return QStringLiteral("best cost %1 %2").arg(double(cost.at<float>(q.y(), q.x())), 0, 'f', 1).arg(unit);
        };

        const Presenter pres(in, intParam(p, "arrows"));
        Output out = pres.present(res.flow, mech, {float(bp.radius), float(bp.search), float(intParam(p, "cost"))});
        // The grey frames, for the cost surface drawn on click.
        out.frames[0].views[kFrame1].data = vl::toGrayF(in.images[0]);
        out.frames[0].views[kFrame2].data = vl::toGrayF(in.images[1]);
        out.stats.push_back({QStringLiteral("Displacements tried per pixel"), QString::number((2 * bp.search + 1) * (2 * bp.search + 1))});
        if (pres.truthMax() > bp.search + 0.5)
            out.note = QStringLiteral("The true motion reaches %1 px, beyond the search radius: those pixels cannot be "
                                      "matched. Raise the search radius.").arg(pres.truthMax(), 0, 'f', 1);
        if (ctx.compare) {
            // DIS, OpenCV's patch-based method: coarse-to-fine patch search plus a variational refinement.
            const auto baseline = vl::flow::opencv::dis(in.images[0], in.images[1]);
            ExtraComparison gross;
            if (pres.hasTruth())
                gross = Comparison{QStringLiteral("Pixels with error > 3 px"), 100 * vl::flow::evaluateFlow(res.flow, in.truth[0]).over3,
                                   100 * vl::flow::evaluateFlow(baseline.flow, in.truth[0]).over3, QStringLiteral(" %"), 1};
            addComparison(out, in, pres, res.flow, oursMs, baseline.flow, baseline.ms, QStringLiteral("DISOpticalFlow, PRESET_MEDIUM"), gross);
        }
        return out;
    }

    // Click a pixel: its block in Frame 1, the search window in Frame 2 with one dot per displacement
    // coloured by its cost (the cost surface), and the winner.
    std::vector<ClickShapes> onClick(const Output& out, int, int, QPointF px) const override {
        auto s = clickShapes(out, px);
        if (s.empty()) return s;
        const auto& views = out.frames.front().views;
        const cv::Mat& I0 = views[kFrame1].data;
        const cv::Mat& I1 = views[kFrame2].data;
        if (I0.empty() || I1.empty()) return s;
        vl::flow::BlockFlowParams bp;
        bp.radius = int(meta(out, kExtra0));
        bp.search = int(meta(out, kExtra1));
        bp.cost = static_cast<vl::flow::BlockCost>(int(meta(out, kExtra2)));
        const cv::Point p(int(px.x()), int(px.y()));
        const cv::Mat surf = vl::flow::blockCostSurface(I0, I1, p, bp);

        // Colour range of the dots: from the best cost to the 90th percentile, at least a few grey levels wide
        // so that a flat surface (all equally good) reads as uniformly blue rather than as noise.
        std::vector<float> v;
        for (auto it = surf.begin<float>(); it != surf.end<float>(); ++it)
            if (std::isfinite(*it)) v.push_back(*it);
        if (v.empty()) return s;
        const double lo = *std::min_element(v.begin(), v.end());
        std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(v.size() * 9 / 10), v.end());
        const double hi = std::max(lo + 4.0, double(v[v.size() * 9 / 10]));
        const auto colour = [&](double c) { return turbo(0.15 + 0.85 * std::clamp((c - lo) / (hi - lo), 0.0, 1.0), 0, 1); };

        const double r = bp.radius, S = bp.search;
        auto& f1 = s[kFrame1].shapes;
        f1.push_back(RectShape{QRectF(p.x - r, p.y - r, 2 * r + 1, 2 * r + 1), {colors::accent, 1.2, false}});
        auto& f2 = s[kFrame2].shapes;
        f2.insert(f2.begin(), RectShape{QRectF(p.x - S, p.y - S, 2 * S + 1, 2 * S + 1), {colors::info, 1.0, true}});
        // One dot per displacement, coloured by its cost: blue = good fit, red = bad.
        std::vector<Shape> dots;
        for (int dy = 0; dy < surf.rows; ++dy)
            for (int dx = 0; dx < surf.cols; ++dx) {
                const float c = surf.at<float>(dy, dx);
                if (!std::isfinite(c)) continue;
                dots.push_back(PointsShape{QPolygonF{QPointF(p.x + dx - S + 0.5, p.y + dy - S + 0.5)},
                                           colour(c), 2.2});
            }
        f2.insert(f2.begin() + 1, dots.begin(), dots.end());
        // The winner: the block the flow points to.
        const cv::Vec2f w = views[kFlow].data.at<cv::Vec2f>(p.y, p.x);
        const QPointF b(p.x + std::round(double(w[0])), p.y + std::round(double(w[1])));
        f2.push_back(RectShape{QRectF(b.x() - r, b.y() - r, 2 * r + 1, 2 * r + 1), {colors::accent, 1.2, false}});
        f2.push_back(LabelShape{QPointF(p.x - S, p.y - S - 20), QStringLiteral("search window · dots: cost of each displacement"),
                                colors::info});
        return s;
    }
};

}  // namespace

VL_REGISTER_MODULE(BlockFlowModule);

}  // namespace vl::app
