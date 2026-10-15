// Rectifying the page with a homography and inverse warping. Course "3D Vision and Motion", part 2.
//
//   1. outputSize      W = max(‖TR − TL‖, ‖BR − BL‖), H = max(‖BL − TL‖, ‖BR − TR‖)
//   2. pageHomography  the DLT on the four correspondences corner_i ↔ rectangle_i
//   3. warpInverse     out(u, v) = photo(H⁻¹ (u, v)), bilinear
//
// The side lengths only approximate the page's true proportions: a side seen at a slant looks
// shorter. Taking the longer of each pair errs towards keeping detail; the A4 option fixes the
// ratio when the user knows the paper.

#include "core/scanner/rectify.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/filters.hpp"
#include "core/geometry/homography.hpp"

namespace vl::scanner {

cv::Size outputSize(const Quad& c, PageRatio ratio, int maxSide) {
    auto length = [](cv::Point2d a, cv::Point2d b) { return std::hypot(a.x - b.x, a.y - b.y); };
    double W = std::max(length(c[0], c[1]), length(c[3], c[2]));
    double H = std::max(length(c[0], c[3]), length(c[1], c[2]));
    if (ratio == PageRatio::A4) {
        // Keep the longer measured side and give the other one the ratio 1 : √2 of ISO 216 paper.
        if (H >= W)
            W = H / std::sqrt(2.0);
        else
            H = W / std::sqrt(2.0);
    }
    const double scale = std::min(1.0, maxSide / std::max(W, H));
    return {std::max(16, int(std::lround(W * scale))), std::max(16, int(std::lround(H * scale)))};
}

Quad rectangleCorners(cv::Size s) {
    const double W = s.width, H = s.height;
    return {cv::Point2d(-0.5, -0.5), cv::Point2d(W - 0.5, -0.5), cv::Point2d(W - 0.5, H - 0.5),
            cv::Point2d(-0.5, H - 0.5)};
}

cv::Matx33d pageHomography(const Quad& corners, cv::Size size) {
    const Quad target = rectangleCorners(size);
    const std::vector<cv::Point2d> x(corners.begin(), corners.end()), xp(target.begin(), target.end());
    const auto H = vl::geometry::dltHomography(x, xp);
    CV_Assert(H.has_value());  // only fails when three corners are collinear, which isPlausible excludes
    return *H;
}

cv::Mat warpInverse(const cv::Mat& photo, const cv::Matx33d& H, cv::Size size) {
    CV_Assert(photo.depth() == CV_8U && (photo.channels() == 1 || photo.channels() == 3));
    const cv::Matx33d Hinv = H.inv();
    // One float image per channel, as vl::bilinear expects.
    std::vector<cv::Mat> channels;
    cv::split(photo, channels);
    for (cv::Mat& ch : channels) ch.convertTo(ch, CV_32F);
    std::vector<cv::Mat> out(channels.size());
    for (cv::Mat& o : out) o.create(size, CV_32F);
    for (int v = 0; v < size.height; ++v)
        for (int u = 0; u < size.width; ++u) {
            // (x, y, w)ᵀ = H⁻¹ (u, v, 1)ᵀ, then divide by w: the projective division.
            const double w = Hinv(2, 0) * u + Hinv(2, 1) * v + Hinv(2, 2);
            const double x = (Hinv(0, 0) * u + Hinv(0, 1) * v + Hinv(0, 2)) / w;
            const double y = (Hinv(1, 0) * u + Hinv(1, 1) * v + Hinv(1, 2)) / w;
            for (size_t k = 0; k < channels.size(); ++k)
                out[k].at<float>(v, u) = vl::bilinear(channels[k], float(x), float(y));
        }
    cv::Mat merged, result;
    cv::merge(out, merged);
    merged.convertTo(result, photo.type());  // rounds and saturates to 8 bits
    return result;
}

cv::Mat rectify(const cv::Mat& photo, const Quad& corners, cv::Size size) {
    return warpInverse(photo, pageHomography(corners, size), size);
}

}  // namespace vl::scanner
