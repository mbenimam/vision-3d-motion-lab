#include "core/scanner/opencv_baselines.hpp"

#include <algorithm>
#include <vector>

#include "core/common/cv_compat.hpp"
#include "core/common/task.hpp"

namespace vl::scanner::opencv {
namespace {

// The largest contour that approxPolyDP (tolerance 2 % of its perimeter) reduces to a convex
// quadrilateral, among the five largest, as in the usual tutorial.
bool largestQuadContour(const cv::Mat& edges, std::vector<cv::Point>& quad) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(),
              [](const auto& a, const auto& b) { return cv::contourArea(a) > cv::contourArea(b); });
    for (size_t i = 0; i < std::min<size_t>(5, contours.size()); ++i) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, 0.02 * cv::arcLength(contours[i], true), true);
        if (approx.size() == 4 && cv::isContourConvex(approx)) {
            quad = approx;
            return true;
        }
    }
    return false;
}

// Background division with OpenCV primitives: the paper's level from a dilation (removes the ink)
// and a median blur, then 255 − |I − background|, stretched to the full range.
cv::Mat flattenChannel(const cv::Mat& ch) {
    cv::Mat background, diff, out;
    cv::dilate(ch, background, cv::getStructuringElement(cv::MORPH_RECT, {7, 7}));
    cv::medianBlur(background, background, 21);
    cv::absdiff(ch, background, diff);
    diff = 255 - diff;
    cv::normalize(diff, out, 0, 255, cv::NORM_MINMAX, CV_8U);
    return out;
}

cv::Mat scanLook(const cv::Mat& page, const CleanupParams& p) {
    if (p.mode == ScanMode::Colour) {
        std::vector<cv::Mat> channels;
        cv::split(page, channels);
        for (cv::Mat& ch : channels) ch = flattenChannel(ch);
        cv::Mat out;
        cv::merge(channels, out);
        return out;
    }
    cv::Mat grey, out;
    cv::cvtColor(page, grey, cv::COLOR_BGR2GRAY);
    if (p.mode == ScanMode::Grey) return flattenChannel(grey);
    cv::adaptiveThreshold(grey, out, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 2 * p.window + 1, p.C);
    return out;
}

}  // namespace

cv::Mat rectify(const cv::Mat& photo, const Quad& corners, cv::Size size) {
    std::vector<cv::Point2f> src, dst;
    const Quad target = {cv::Point2d(-0.5, -0.5), cv::Point2d(size.width - 0.5, -0.5),
                         cv::Point2d(size.width - 0.5, size.height - 0.5), cv::Point2d(-0.5, size.height - 0.5)};
    for (size_t i = 0; i < 4; ++i) {
        src.emplace_back(corners[i]);
        dst.emplace_back(target[i]);
    }
    cv::Mat out;
    cv::warpPerspective(photo, out, cv::getPerspectiveTransform(src, dst), size, cv::INTER_LINEAR,
                        cv::BORDER_REPLICATE);
    return out;
}

Scan documentScan(const cv::Mat& photo, int detectionSide, double cannyLow, double cannyHigh, cv::Size outSize,
                  const CleanupParams& cleanup) {
    Scan s;
    const Stopwatch clock;
    const double scale = std::min(1.0, double(detectionSide) / std::max(photo.cols, photo.rows));
    cv::Mat small, grey, edges;
    cv::resize(photo, small, {}, scale, scale, cv::INTER_AREA);
    cv::cvtColor(small, grey, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grey, grey, {0, 0}, 2.0);
    // cv::Canny's 3×3 Sobel is not normalised: a unit ramp gives 8, ours gives 1.
    cv::Canny(grey, edges, 8 * cannyLow, 8 * cannyHigh, 3, true);
    std::vector<cv::Point> quad;
    s.found = largestQuadContour(edges, quad);
    if (s.found) {
        std::vector<cv::Point2d> pts;
        // Small pixel centre x sits at full-size (x + ½) / scale − ½ (cv::resize aligns pixel edges).
        for (const auto& q : quad) pts.emplace_back((q.x + 0.5) / scale - 0.5, (q.y + 0.5) / scale - 0.5);
        s.corners = orderCorners(pts);
    } else {
        s.corners = {cv::Point2d(0, 0), cv::Point2d(photo.cols - 1, 0), cv::Point2d(photo.cols - 1, photo.rows - 1),
                     cv::Point2d(0, photo.rows - 1)};
    }
    s.rectified = rectify(photo, s.corners, outSize);
    s.scan = scanLook(s.rectified, cleanup);
    s.ms = clock.ms();
    return s;
}

}  // namespace vl::scanner::opencv
