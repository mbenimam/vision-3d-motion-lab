#include "app/modules/flow/flow_views.hpp"

#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

#include "core/common/image.hpp"
#include "core/common/synthetic.hpp"
#include "core/flow/flow_color.hpp"
#include "core/flow/flow_eval.hpp"

namespace vl::app::flow_views {
namespace {

constexpr double kErrorMax = 3.0;  // px, top of the error colour scale

QString vec(cv::Vec2f w) {
    return QStringLiteral("(%1, %2) px").arg(double(w[0]), 0, 'f', 2).arg(double(w[1]), 0, 'f', 2);
}

bool inside(const cv::Mat& m, int x, int y) { return x >= 0 && y >= 0 && x < m.cols && y < m.rows; }

}  // namespace

QPointF labelAt(QPointF p, cv::Size size) {
    // Right of the point in the left half, left of it in the right half, so the text stays on the picture.
    return p.x() < 0.55 * size.width ? p + QPointF(10, 8) : QPointF(p.x() - 0.4 * size.width, p.y() + 10);
}

namespace {

// Text under the cursor on the flow-like views: the estimate, the truth and the error.
std::function<QString(QPoint)> flowProbe(const cv::Mat& flow, const cv::Mat& truth) {
    return [flow, truth](QPoint q) {
        if (!inside(flow, q.x(), q.y())) return QString();
        const cv::Vec2f w = flow.at<cv::Vec2f>(q.y(), q.x());
        QString s = std::isfinite(w[0]) ? QStringLiteral("w = %1  ·  |w| = %2 px").arg(vec(w)).arg(std::hypot(w[0], w[1]), 0, 'f', 2)
                                        : QStringLiteral("no estimate (unreliable)");
        if (!truth.empty()) {
            const cv::Vec2f t = truth.at<cv::Vec2f>(q.y(), q.x());
            if (vl::flow::isKnownFlow(t)) {
                s += QStringLiteral("  ·  truth %1").arg(vec(t));
                if (std::isfinite(w[0])) s += QStringLiteral("  ·  error %1 px").arg(std::hypot(w[0] - t[0], w[1] - t[1]), 0, 'f', 2);
            } else {
                s += QStringLiteral("  ·  truth unknown");
            }
        }
        return s;
    };
}

// Axes, the unit circle and the direction names drawn over the colour wheel.
std::vector<Shape> keyShapes(cv::Size size, float scale) {
    const cv::Point2f c = vl::flow::wheelPosition(size, scale, {0, 0});
    const float r = vl::flow::wheelPosition(size, scale, {scale, 0}).x - c.x;
    const Pen axis{colors::muted, 1.0, true};
    std::vector<Shape> s;
    s.push_back(LineShape{QLineF(c.x - r, c.y, c.x + r, c.y), axis});
    s.push_back(LineShape{QLineF(c.x, c.y - r, c.x, c.y + r), axis});
    s.push_back(EllipseShape{QPointF(c.x, c.y), r, r, 0, {colors::muted, 1.0, false}});
    s.push_back(EllipseShape{QPointF(c.x, c.y), r / 2, r / 2, 0, axis});
    s.push_back(LabelShape{QPointF(c.x + r + 4, c.y - 8), QStringLiteral("+u"), colors::muted});
    s.push_back(LabelShape{QPointF(c.x + 4, c.y + r + 2), QStringLiteral("+v (down)"), colors::muted});
    s.push_back(LabelShape{QPointF(c.x + 0.71 * r + 6, c.y - 0.71 * r - 18),
                           QStringLiteral("|w| = %1 px").arg(double(scale), 0, 'f', 1), colors::muted});
    s.push_back(LabelShape{QPointF(c.x + 0.35 * r + 4, c.y - 0.35 * r - 18),
                           QStringLiteral("%1").arg(double(scale) / 2, 0, 'f', 1), colors::muted});
    return s;
}

}  // namespace

QColor turbo(double v, double lo, double hi) {
    cv::Mat m(1, 1, CV_32F, cv::Scalar(std::isfinite(v) ? v : hi));
    const cv::Vec3b b = vl::colorize(m, lo, hi).at<cv::Vec3b>(0, 0);
    return {b[2], b[1], b[0]};
}

float meta(const Output& out, Meta m) {
    if (out.frames.empty() || int(out.frames[0].views.size()) <= kKey) return 0.f;
    const cv::Mat& d = out.frames[0].views[kKey].data;
    return d.cols > int(m) ? d.at<float>(0, int(m)) : 0.f;
}

Presenter::Presenter(const Inputs& in, int arrowStep) : in_(in), arrowStep_(arrowStep) {
    // Everything that does not depend on the estimate is made once, so partial results stay cheap.
    frame1_ = toQImage(in.images[0]);
    frame2_ = toQImage(in.images[1]);
    const cv::Size size = in.images[0].size();
    if (!in.truth.empty() && in.truth[0].type() == CV_32FC2 && in.truth[0].size() == size) {
        truth_ = in.truth[0];
        truthScale_ = vl::flow::robustMaxMagnitude(truth_);
        cv::Mat shown(size, CV_32FC2);  // unknown vectors as NaN, which the colour code paints black
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (int y = 0; y < size.height; ++y) {
            const auto* t = truth_.ptr<cv::Vec2f>(y);
            auto* o = shown.ptr<cv::Vec2f>(y);
            for (int x = 0; x < size.width; ++x) {
                const bool known = vl::flow::isKnownFlow(t[x]);
                o[x] = known ? t[x] : cv::Vec2f(nan, nan);
                if (known) truthMax_ = std::max(truthMax_, double(std::hypot(t[x][0], t[x][1])));
            }
        }
        truthImage_ = toQImage(vl::flow::flowToColor(shown, truthScale_));
        keyImage_ = toQImage(vl::flow::colorWheel(size, truthScale_));
    }
}

Output Presenter::present(const cv::Mat& flow, View mechanism, const MetaValues& extra) const {
    const cv::Size size = flow.size();
    // One colour scale for estimate, truth and key: the truth's when known, so they compare directly.
    const float scale = hasTruth() ? truthScale_ : vl::flow::robustMaxMagnitude(flow);
    const auto probe = flowProbe(flow, truth_);

    View f1{QStringLiteral("Frame 1"), frame1_};
    f1.probe = probe;
    if (arrowStep_ > 0) {
        const float k = vl::flow::arrowScale(flow, arrowStep_);
        ArrowsShape arrows{{}, {colors::accent, 1.2, false}};
        for (const auto& a : vl::flow::flowArrows(flow, arrowStep_, k))
            arrows.arrows.emplace_back(a.from.x, a.from.y, a.to.x, a.to.y);
        f1.shapes.push_back(std::move(arrows));
        if (k > 1.05f)
            f1.shapes.push_back(LabelShape{QPointF(6, 4), QStringLiteral("arrows drawn %1× longer").arg(double(k), 0, 'f', 1)});
    }

    View f2{QStringLiteral("Frame 2"), frame2_};
    View fl{QStringLiteral("Flow"), toQImage(vl::flow::flowToColor(flow, scale)), {}, probe, flow};
    // The wheel in the corner: direction = hue, length = saturation, rim = the colour scale.
    fl.legend = Legend{QStringLiteral("rim = %1 px").arg(double(scale), 0, 'f', 1), 0, double(scale), {}, {},
                       toQImage(vl::flow::colorWheel({96, 96}, scale, 4))};

    cv::Mat metaRow(1, kMetaSize, CV_32F);
    metaRow.at<float>(0, kScale) = scale;
    for (int i = 0; i < 3; ++i) metaRow.at<float>(0, kExtra0 + i) = extra[size_t(i)];
    View key{QStringLiteral("Colour key"), hasTruth() ? keyImage_ : toQImage(vl::flow::colorWheel(size, scale)),
             keyShapes(size, scale), {}, metaRow};
    key.probe = [size, scale](QPoint q) {
        const cv::Point2f c = vl::flow::wheelPosition(size, scale, {0, 0});
        const float r = vl::flow::wheelPosition(size, scale, {scale, 0}).x - c.x;
        const cv::Vec2f w((float(q.x()) + 0.5f - c.x) / r * scale, (float(q.y()) + 0.5f - c.y) / r * scale);
        return QStringLiteral("this colour means w = %1").arg(vec(w));
    };

    cv::Mat finite = flow.clone();
    cv::patchNaNs(finite, 0.0);
    const cv::Mat residual = vl::flow::brightnessResidual(in_.images[0], in_.images[1], flow);
    View warped{QStringLiteral("Warped"), toQImage(vl::synth::warpBackward(in_.images[1], finite))};
    warped.probe = [residual](QPoint q) {
        if (!inside(residual, q.x(), q.y())) return QString();
        const float r = residual.at<float>(q.y(), q.x());
        return std::isfinite(r) ? QStringLiteral("|I₂(x + w) − I₁(x)| = %1 grey levels").arg(double(r), 0, 'f', 1)
                                : QStringLiteral("no estimate, or warped from outside frame 2");
    };

    Frame fr;
    fr.views = {f1, f2, mechanism, fl, key, warped};
    Output out;
    if (hasTruth()) {
        View tv{QStringLiteral("Truth"), truthImage_, {}, probe, truth_};
        const cv::Mat epe = vl::flow::endpointErrorMap(flow, truth_);
        cv::Mat valid = epe == epe;  // false on NaN
        View ev{QStringLiteral("Error"), toQImage(vl::colorize(epe, 0, kErrorMax, valid)), {}, probe, epe,
                Legend{QStringLiteral("endpoint error |w − w_gt| (px) · black: unknown"), 0, kErrorMax, QStringLiteral("0"),
                       QStringLiteral("≥ 3")}};
        fr.views.push_back(tv);
        fr.views.push_back(ev);
        const auto e = vl::flow::evaluateFlow(flow, truth_);
        out.stats.push_back({QStringLiteral("Average endpoint error"), fmt(e.epe, 3) + QStringLiteral(" px"), true});
        out.stats.push_back({QStringLiteral("Average angular error"), fmt(e.angular, 2) + QStringLiteral("°")});
        out.stats.push_back({QStringLiteral("Pixels with error > 1 px"), fmt(100 * e.over1) + QStringLiteral(" %")});
        if (e.missing > 0)
            out.stats.push_back({QStringLiteral("Pixels without estimate"),
                                 fmt(100.0 * double(e.missing) / double(e.missing + e.count)) + QStringLiteral(" %")});
    } else {
        double sum = 0;
        long n = 0;
        for (int y = 0; y < residual.rows; ++y)
            for (int x = 0; x < residual.cols; ++x)
                if (const float r = residual.at<float>(y, x); std::isfinite(r)) {
                    sum += r;
                    ++n;
                }
        out.stats.push_back({QStringLiteral("Mean residual |I₂(x + w) − I₁|"), fmt(n ? sum / double(n) : 0.0, 2) + QStringLiteral(" grey levels"), true});
    }
    out.frames.push_back(std::move(fr));
    return out;
}

std::vector<Module::ClickShapes> clickShapes(const Output& out, QPointF pixel) {
    if (out.frames.empty()) return {};
    const auto& views = out.frames.front().views;
    const cv::Mat& flow = views[kFlow].data;
    const int x = int(pixel.x()), y = int(pixel.y());
    if (!inside(flow, x, y)) return {};
    const cv::Vec2f w = flow.at<cv::Vec2f>(y, x);
    const bool ok = std::isfinite(w[0]) && std::isfinite(w[1]);
    const cv::Mat truth = views.size() > size_t(kTruth) ? views[kTruth].data : cv::Mat();
    const bool hasT = !truth.empty() && vl::flow::isKnownFlow(truth.at<cv::Vec2f>(y, x));
    const cv::Vec2f t = hasT ? truth.at<cv::Vec2f>(y, x) : cv::Vec2f();
    const QPointF p(x + 0.5, y + 0.5);
    const QPointF pw = p + QPointF(w[0], w[1]), pt = p + QPointF(t[0], t[1]);

    std::vector<Module::ClickShapes> s(views.size());
    for (size_t v = 0; v < views.size(); ++v) {
        s[v].view = int(v);
        if (int(v) == kKey || int(v) == kFrame2) continue;
        // The vector at its true length (a pixel or two), with a ring to find it at low zoom.
        s[v].shapes.push_back(EllipseShape{p, 7, 7, 0, {colors::accent, 1.2, false}});
        if (hasT) s[v].shapes.push_back(ArrowsShape{{QLineF(p, pt)}, {colors::good, 1.4, false}});
        if (ok) s[v].shapes.push_back(ArrowsShape{{QLineF(p, pw)}, {colors::accent, 1.6, false}});
    }
    // Frame 2: where the pixel was, where the flow sends it, where it really went.
    auto& f2 = s[kFrame2].shapes;
    f2.push_back(PointsShape{QPolygonF{p}, colors::muted, 3});
    if (hasT) f2.push_back(PointsShape{QPolygonF{pt}, colors::good, 4});
    if (ok) {
        f2.push_back(LineShape{QLineF(p, pw), {colors::accent, 1.0, true}});
        f2.push_back(PointsShape{QPolygonF{pw}, colors::accent, 4});
    }
    QString text = ok ? QStringLiteral("w = %1").arg(vec(w)) : QStringLiteral("no estimate");
    if (hasT) text += QStringLiteral("   truth %1").arg(vec(t));
    f2.push_back(LabelShape{labelAt(p, flow.size()), text});

    // Colour key: the vector's place on the wheel.
    const float scale = meta(out, kScale);
    const cv::Size size = flow.size();
    const cv::Point2f c = vl::flow::wheelPosition(size, scale, {0, 0});
    auto& k = s[kKey].shapes;
    if (hasT) {
        const cv::Point2f q = vl::flow::wheelPosition(size, scale, {t[0], t[1]});
        k.push_back(PointsShape{QPolygonF{QPointF(q.x, q.y)}, colors::good, 5});
    }
    if (ok) {
        const cv::Point2f q = vl::flow::wheelPosition(size, scale, {w[0], w[1]});
        k.push_back(ArrowsShape{{QLineF(c.x, c.y, q.x, q.y)}, {colors::accent, 1.6, false}});
        k.push_back(LabelShape{labelAt(QPointF(q.x, q.y), size), text});
    }
    return s;
}

}  // namespace vl::app::flow_views
