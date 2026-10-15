// Topic: dense Lucas–Kanade, a local least-squares flow, with the aperture problem made visible and
// the pyramid (coarse to fine) that lets it follow motions larger than a pixel.

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include "app/module.hpp"
#include "app/modules/flow/flow_samples.hpp"
#include "app/modules/flow/flow_views.hpp"
#include "core/common/image.hpp"
#include "core/flow/flow_eval.hpp"
#include "core/flow/lucas_kanade.hpp"
#include "core/flow/opencv_baselines.hpp"

namespace vl::app {
namespace {

using namespace flow_views;

// What the two eigenvalues of the structure tensor say about a window; `ref` is a typical strong λ₁
// of the image (contrast differs a lot between images, so "small" is relative).
QString windowKind(double l1, double l2, double ref) {
    if (l1 < 0.02 * ref) return QStringLiteral("flat: the window says nothing");
    if (l2 < 0.15 * l1) return QStringLiteral("edge: only the motion across it is measurable");
    return QStringLiteral("texture or corner: both components pinned down");
}

double percentile(const cv::Mat& m, double q) {
    std::vector<float> v(m.begin<float>(), m.end<float>());
    const auto k = std::ptrdiff_t(q * double(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[size_t(k)];
}

class LucasKanadeModule final : public Module {
public:
    Info info() const override {
        return {QStringLiteral("flow-lk"), QStringLiteral("Lucas–Kanade"), QStringLiteral("Optical flow"),
                QStringLiteral("One pixel gives one equation, I<sub>x</sub>u + I<sub>y</sub>v + I<sub>t</sub> = 0, "
                               "for two unknowns. Lucas–Kanade assumes the flow is constant over a small Gaussian "
                               "window and solves the 2×2 least-squares system there. Its <i>structure tensor</i> "
                               "tells how well posed that is: on a straight edge only the motion across the edge "
                               "is seen, the <b>aperture problem</b>."),
                QStringLiteral("turn <i>Coarse to fine</i> off on <i>Synthetic: rotation and zoom</i>: the corners, "
                               "which move 11 px, fall apart. Then click the edge of the <i>flat square</i>."),
                InputKind::ImagePair, 710, QStringLiteral("Frame 1"), QStringLiteral("cv::calcOpticalFlowPyrLK"),
                {QStringLiteral("src/core/flow/lucas_kanade.hpp"), QStringLiteral("src/core/flow/lucas_kanade.cpp"),
                 QStringLiteral("src/core/flow/flow_common.hpp"), QStringLiteral("src/core/flow/flow_common.cpp"),
                 QStringLiteral("src/core/common/filters.hpp"), QStringLiteral("src/core/common/filters.cpp")}};
    }

    std::vector<ParamSpec> params() const override {
        return {
            ParamSpec::real(QStringLiteral("sigma"), QStringLiteral("Window σ"), 1.0, 6.0, 2.5, 0.5, 1, QStringLiteral(" px"))
                .withHelp(QStringLiteral("Gaussian weight of the neighbours that share the pixel's motion. Larger: "
                                         "fewer aperture failures, blurrier motion boundaries.")),
            ParamSpec::toggle(QStringLiteral("c2f"), QStringLiteral("Coarse to fine"), true)
                .withHelp(QStringLiteral("Estimate on halved copies first, where the motion is smaller, then refine. "
                                         "Off: a single scale, which cannot follow more than a pixel or two.")),
            ParamSpec::integer(QStringLiteral("levels"), QStringLiteral("Pyramid levels"), 2, 6, 4)
                .withHelp(QStringLiteral("Each level halves the image, so L levels reach motions about 2^(L−1) times larger.")),
            ParamSpec::integer(QStringLiteral("iterations"), QStringLiteral("Warping iterations per level"), 1, 10, 3)
                .withHelp(QStringLiteral("Re-warp frame 2 by the current flow and solve again: Gauss–Newton steps on the "
                                         "non-linear brightness constancy.")),
            ParamSpec::real(QStringLiteral("minEigen"), QStringLiteral("Reject windows with λ₂ below"), 0, 100, 0, 5, 0,
                            QStringLiteral(" grey²/px²"))
                .withHelp(QStringLiteral("Bouguet's reliability test: no estimate where the smallest eigenvalue is "
                                         "small (flat areas and edges). 0 keeps every pixel.")),
            ParamSpec::integer(QStringLiteral("arrows"), QStringLiteral("Arrow spacing"), 8, 48, 16, 4, QStringLiteral(" px")),
        };
    }

    std::vector<Sample> samples() const override { return flow_samples::all(); }

    Output compute(const Inputs& in, const Params& p, const RunContext& ctx) const override {
        vl::flow::LucasKanadeParams lp;
        lp.sigma = realParam(p, "sigma");
        const bool c2f = toggleParam(p, "c2f");
        lp.levels = c2f ? intParam(p, "levels") : 1;
        lp.iterations = intParam(p, "iterations");
        lp.minEigen = realParam(p, "minEigen");
        const vl::Stopwatch clock;
        const auto res = vl::flow::lucasKanade(in.images[0], in.images[1], lp, ctx.stop, ctx.progress);
        const double oursMs = clock.ms();

        // Mechanism: the smallest eigenvalue of the structure tensor, on a log scale up to the image's 99th percentile.
        cv::Mat logL2;
        cv::log(res.minEigen + 1.0, logL2);
        logL2 /= std::log(10.0);
        const double top = std::max(1.0, percentile(logL2, 0.99));
        const double ref = percentile(res.maxEigen, 0.95);
        const cv::Mat l1 = res.maxEigen, l2 = res.minEigen;
        View conf{QStringLiteral("Confidence"), toQImage(vl::colorize(logL2, 0, top)), {}, {}, res.tensor,
                  Legend{QStringLiteral("smallest eigenvalue λ₂ of the structure tensor (grey²/px², log scale)"), 0, top,
                         QStringLiteral("0: flat or edge"), QStringLiteral("%1: texture").arg(std::pow(10.0, top) - 1, 0, 'f', 0)}};
        conf.probe = [l1, l2, ref](QPoint q) {
            if (q.x() < 0 || q.y() < 0 || q.x() >= l1.cols || q.y() >= l1.rows) return QString();
            const double a = l1.at<float>(q.y(), q.x()), b = l2.at<float>(q.y(), q.x());
            return QStringLiteral("λ₁ = %1, λ₂ = %2  ·  %3").arg(a, 0, 'f', 1).arg(b, 0, 'f', 1).arg(windowKind(a, b, ref));
        };

        const Presenter pres(in, intParam(p, "arrows"));
        Output out = pres.present(res.flow, conf, {float(lp.sigma), float(ref), 0.f});
        out.stats.push_back({QStringLiteral("Pyramid levels used"), QString::number(res.levelsUsed)});
        out.series.push_back({QStringLiteral("Mean update per iteration (px)"), res.meanUpdate});
        if (!c2f && pres.truthMax() > 2.5)
            out.note = QStringLiteral("The true motion reaches %1 px; one scale only linearises about a pixel or two. "
                                      "Turn coarse to fine on.").arg(pres.truthMax(), 0, 'f', 1);
        if (ctx.compare) compareWithOpenCv(in, lp, pres, res.flow, oursMs, out);
        return out;
    }

    // OpenCV's pyramidal LK at every pixel (one tracked point per pixel), with the same pyramid and
    // a square window of about the same reach as our Gaussian one (±2σ holds 95 % of its weight).
    static void compareWithOpenCv(const Inputs& in, const vl::flow::LucasKanadeParams& lp, const Presenter& pres,
                                  const cv::Mat& ours, double oursMs, Output& out) {
        const int radius = int(std::lround(2 * lp.sigma));
        const auto baseline = vl::flow::opencv::pyramidalLucasKanade(in.images[0], in.images[1], radius, lp.levels);
        const Comparison coverage{QStringLiteral("Pixels with an estimate"), coveragePercent(ours), coveragePercent(baseline.flow),
                                  QStringLiteral(" %"), 1, false};
        addComparison(out, in, pres, ours, oursMs, baseline.flow, baseline.ms,
                      QStringLiteral("calcOpticalFlowPyrLK, %1×%1 window").arg(2 * radius + 1), coverage);
    }

    // Click a pixel: its window, and the ellipse of displacements that fit the window about equally
    // well (axes ∝ 1/√λ): round on texture, a long needle along an edge (the aperture problem).
    std::vector<ClickShapes> onClick(const Output& out, int, int, QPointF px) const override {
        auto s = clickShapes(out, px);
        if (s.empty()) return s;
        const cv::Mat& J = out.frames.front().views[kMechanism].data;
        const int x = int(px.x()), y = int(px.y());
        const cv::Vec3f t = J.at<cv::Vec3f>(y, x);
        const double a = t[0], b = t[1], c = t[2];
        const double m = 0.5 * (a + c), r = std::sqrt(0.25 * (a - c) * (a - c) + b * b);
        const double l1 = m + r, l2 = std::max(0.0, m - r);
        const double theta = 0.5 * std::atan2(2 * b, a - c) * 180.0 / std::numbers::pi;  // direction of λ₁ (the gradient)
        const double sigma = meta(out, kExtra0), ref = meta(out, kExtra1);
        // A window as strong as the image's typical texture gets a circle of the window's radius.
        const auto axis = [&](double l) { return std::min(60.0, 2 * sigma * std::sqrt(ref / std::max(l, 1e-6))); };
        const QPointF p(x + 0.5, y + 0.5);
        const cv::Vec2f w = out.frames.front().views[kFlow].data.at<cv::Vec2f>(y, x);
        const QPointF pw = std::isfinite(w[0]) ? p + QPointF(w[0], w[1]) : p;

        for (int v : {int(kFrame1), int(kMechanism)}) {
            s[size_t(v)].shapes.push_back(EllipseShape{p, 2 * sigma, 2 * sigma, 0, {colors::info, 1.0, true}});
        }
        const QString kind = windowKind(l1, l2, ref).section(QLatin1Char(':'), 0, 0);
        s[kMechanism].shapes.push_back(LabelShape{labelAt(p, J.size()),
                                                 QStringLiteral("λ₁ %1 · λ₂ %2 · %3").arg(l1, 0, 'f', 0).arg(l2, 0, 'f', 1).arg(kind)});
        auto& f2 = s[kFrame2].shapes;
        f2.insert(f2.begin(), EllipseShape{pw, axis(l1), axis(l2), theta, {colors::info, 1.4, false}});
        return s;
    }
};

}  // namespace

VL_REGISTER_MODULE(LucasKanadeModule);

}  // namespace vl::app
