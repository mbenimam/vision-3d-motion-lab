#include "app/modules/stereo/stereo_views.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

#include "app/modules/stereo/stereo_samples.hpp"
#include "core/common/image.hpp"
#include "core/stereo/block_matching.hpp"
#include "core/stereo/graphcut_stereo.hpp"

namespace vl::app::stereo_views {

Working prepare(const Inputs& in, int factor) {
    Working w;
    w.factor = std::max(1, factor);
    w.left = vl::toBGR(in.images[0]);
    w.right = vl::toBGR(in.images[1]);
    const bool hasTruth = !in.truth.empty() && !in.truth[0].empty();
    if (hasTruth) w.truth = in.truth[0];
    if (w.factor > 1) {
        const double s = 1.0 / w.factor;
        cv::resize(w.left, w.left, cv::Size(), s, s, cv::INTER_AREA);
        cv::resize(w.right, w.right, cv::Size(), s, s, cv::INTER_AREA);
        if (hasTruth) {
            cv::resize(w.truth, w.truth, w.left.size(), 0, 0, cv::INTER_NEAREST);
            w.truth = w.truth / w.factor;
        }
    }
    if (hasTruth) {
        if (in.truth.size() > 1 && !in.truth[1].empty())
            cv::resize(in.truth[1], w.visible, w.left.size(), 0, 0, cv::INTER_NEAREST);
        else
            cv::bitwise_not(vl::stereo::occlusionsFromDisparity(w.truth), w.visible);  // no mask given: derive it
    }
    w.maxDisparity = int(std::ceil(double(stereo_samples::maxDisparityHint(in)) / w.factor));
    return w;
}

View disparityView(const QString& name, const cv::Mat& disparity, const Working& w, const QString& nanText) {
    cv::Mat valid(disparity.size(), CV_8U), shown(disparity.size(), CV_32F);
    for (int y = 0; y < disparity.rows; ++y)
        for (int x = 0; x < disparity.cols; ++x) {
            const float d = disparity.at<float>(y, x);
            valid.at<uchar>(y, x) = std::isfinite(d) ? 255 : 0;
            shown.at<float>(y, x) = std::isfinite(d) ? d : 0.f;
        }
    const double hi = w.maxDisparity;
    View v{name, toQImage(vl::colorize(shown, 0, hi, valid)), {}, {}, disparity,
           Legend{QStringLiteral("disparity (px) · black: %1").arg(nanText), 0, hi, {}, {}}};
    const cv::Mat d = disparity, truth = w.truth;
    v.probe = [d, truth, nanText](QPoint q) {
        if (q.x() < 0 || q.y() < 0 || q.x() >= d.cols || q.y() >= d.rows) return QString();
        const float dv = d.at<float>(q.y(), q.x());
        QString s = std::isfinite(dv) ? QStringLiteral("d = %1 px").arg(dv, 0, 'f', 0) : nanText;
        if (!truth.empty() && truth.at<float>(q.y(), q.x()) > 0)
            s += QStringLiteral("  ·  truth %1 px").arg(truth.at<float>(q.y(), q.x()), 0, 'f', 1);
        return s;
    };
    return v;
}

double badRate(const cv::Mat& disparity, const Working& w) {
    return w.truth.empty() ? 0.0 : vl::stereo::badPixelRate(disparity, w.truth, w.visible, 1.0);
}

cv::Mat dimmed(const cv::Mat& bgr) {
    cv::Mat g, out;
    cv::cvtColor(bgr, g, cv::COLOR_BGR2GRAY);
    cv::cvtColor(g / 2, out, cv::COLOR_GRAY2BGR);
    return out;
}

cv::Mat errorImage(const cv::Mat& disparity, const Working& w) {
    cv::Mat err = dimmed(w.left);
    for (int y = 0; y < disparity.rows; ++y)
        for (int x = 0; x < disparity.cols; ++x) {
            const float t = w.truth.at<float>(y, x);
            if (t <= 0 || !w.visible.at<uchar>(y, x)) continue;
            const float v = disparity.at<float>(y, x);
            if (!std::isfinite(v))
                err.at<cv::Vec3b>(y, x) = {247, 195, 79};  // colors::info: no answer
            else if (std::abs(v - t) > 1.0f)
                err.at<cv::Vec3b>(y, x) = {138, 92, 255};  // colors::bad: wrong answer
        }
    return err;
}

std::vector<Module::ClickShapes> matchClick(const cv::Mat& disp, int rightView, int viewCount, QPointF px,
                                            const QString& noMatchText) {
    const int x = int(px.x()), y = int(px.y());
    if (disp.empty() || x < 0 || y < 0 || x >= disp.cols || y >= disp.rows) return {};
    const double d = disp.at<float>(y, x);
    const double r = 3;
    std::vector<Module::ClickShapes> s(static_cast<size_t>(viewCount));
    for (int v = 0; v < viewCount; ++v) {
        s[size_t(v)].view = v;
        if (v != rightView) s[size_t(v)].shapes.push_back(RectShape{QRectF(x - r, y - r, 2 * r + 1, 2 * r + 1), {}});
    }
    auto& right = s[size_t(rightView)].shapes;
    right.push_back(LineShape{QLineF(0, y + 0.5, disp.cols, y + 0.5), {colors::info, 1.0, true}});
    right.push_back(LineShape{QLineF(x + 0.5, y - 8, x + 0.5, y + 9), {colors::muted, 1.0, true}});
    if (std::isfinite(d)) {
        right.push_back(RectShape{QRectF(x - d - r, y - r, 2 * r + 1, 2 * r + 1), {}});
        right.push_back(LabelShape{QPointF(x - d + r + 3, y + 6), QStringLiteral("match: %1 px to the left").arg(d, 0, 'f', 0)});
    } else {
        right.push_back(LabelShape{QPointF(x + 4, y + 6), noMatchText, colors::bad});
    }
    return s;
}

}  // namespace vl::app::stereo_views
