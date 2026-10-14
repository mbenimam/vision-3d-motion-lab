// Panorama stitching: chained homographies, inverse-mapping warp, feather blending. Course "3D Vision
// and Motion", part 2. The steps are listed in stitch.hpp.

#include "core/geometry/stitch.hpp"

#include <algorithm>
#include <cmath>

#include "core/common/filters.hpp"
#include "core/common/image.hpp"
#include "core/common/task.hpp"
#include "core/geometry/homography.hpp"

namespace vl::geometry {

std::vector<cv::Matx33d> chainToReference(const std::vector<cv::Matx33d>& links, int reference) {
    const int n = int(links.size()) + 1;
    CV_Assert(reference >= 0 && reference < n);
    std::vector<cv::Matx33d> toRef(size_t(n), cv::Matx33d::eye());
    // Left of the reference: image i -> i+1 -> ... -> r.
    for (int i = reference - 1; i >= 0; --i) toRef[size_t(i)] = toRef[size_t(i + 1)] * links[size_t(i)];
    // Right of it: image i -> i−1 -> ... -> r, each link inverted.
    for (int i = reference + 1; i < n; ++i) toRef[size_t(i)] = toRef[size_t(i - 1)] * links[size_t(i - 1)].inv();
    for (auto& H : toRef)
        if (std::abs(H(2, 2)) > 1e-12) H *= 1.0 / H(2, 2);
    return toRef;
}

std::vector<cv::Point2d> warpedOutline(cv::Size size, const cv::Matx33d& H, int perEdge) {
    const double W = size.width, Hh = size.height;
    const cv::Point2d corner[] = {{0, 0}, {W, 0}, {W, Hh}, {0, Hh}};
    std::vector<cv::Point2d> out;
    for (int e = 0; e < 4; ++e)
        for (int k = 0; k < perEdge; ++k) {
            const double t = double(k) / perEdge;
            out.push_back(transform(H, corner[e] + (corner[(e + 1) % 4] - corner[e]) * t));
        }
    return out;
}

namespace {

// Bounding box of a set of points, ignoring non-finite ones (a corner behind the camera).
cv::Rect2d boundingBox(const std::vector<cv::Point2d>& pts) {
    double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
    for (const auto& q : pts) {
        if (!std::isfinite(q.x) || !std::isfinite(q.y)) continue;
        x0 = std::min(x0, q.x), y0 = std::min(y0, q.y), x1 = std::max(x1, q.x), y1 = std::max(y1, q.y);
    }
    if (x1 < x0) return {};
    return {x0, y0, x1 - x0, y1 - y0};
}

// Step 4: distance of (u, v) to the nearest border of a w × h image, relative to half its smaller
// side, in (0, 1]. The +1 keeps the very edge from weighing exactly 0.
float featherWeight(double u, double v, int w, int h) {
    const double d = std::min({u, v, w - 1 - u, h - 1 - v}) + 1;
    return float(std::min(d / (0.5 * std::min(w, h)), 1.0));
}

// The B, G, R planes of an 8-bit image as float images (what vl::bilinear reads).
std::vector<cv::Mat> floatChannels(const cv::Mat& image) {
    cv::Mat bgr;
    toBGR(image).convertTo(bgr, CV_32FC3);
    std::vector<cv::Mat> planes;
    cv::split(bgr, planes);
    return planes;
}

// Sums of weighted colours and of weights, one per canvas pixel.
struct Accumulator {
    cv::Mat sum;     // CV_32FC3
    cv::Mat weight;  // CV_32F
};

// Step 3. Only the canvas pixels inside the image's warped box are visited.
void warpInto(Accumulator& acc, const cv::Mat& image, const cv::Matx33d& toCanvas, bool feather) {
    const std::vector<cv::Mat> channel = floatChannels(image);
    const cv::Matx33d toImage = toCanvas.inv();
    const cv::Rect2d box = boundingBox(warpedOutline(image.size(), toCanvas));
    const int x0 = std::max(0, int(std::floor(box.x))), x1 = std::min(acc.weight.cols, int(std::ceil(box.br().x)) + 1);
    const int y0 = std::max(0, int(std::floor(box.y))), y1 = std::min(acc.weight.rows, int(std::ceil(box.br().y)) + 1);
    const int w = image.cols, h = image.rows;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) {
            // x_image = (T H)⁻¹ p, homogeneous; w ≤ 0 is behind the camera.
            const cv::Vec3d q = toImage * cv::Vec3d(x, y, 1);
            if (q[2] <= 0) continue;
            const double u = q[0] / q[2], v = q[1] / q[2];
            if (u < 0 || v < 0 || u > w - 1 || v > h - 1) continue;
            const cv::Vec3f colour(bilinear(channel[0], float(u), float(v)), bilinear(channel[1], float(u), float(v)),
                                   bilinear(channel[2], float(u), float(v)));
            if (feather) {
                const float weight = featherWeight(u, v, w, h);
                acc.sum.at<cv::Vec3f>(y, x) += weight * colour;
                acc.weight.at<float>(y, x) += weight;
            } else {  // pasted over whatever was there
                acc.sum.at<cv::Vec3f>(y, x) = colour;
                acc.weight.at<float>(y, x) = 1.f;
            }
        }
}

// Step 5: Σ w c / Σ w, black where no image reaches.
cv::Mat blend(const Accumulator& acc) {
    cv::Mat out(acc.weight.size(), CV_8UC3, cv::Scalar::all(0));
    for (int y = 0; y < out.rows; ++y)
        for (int x = 0; x < out.cols; ++x) {
            const float w = acc.weight.at<float>(y, x);
            if (w <= 0) continue;
            const cv::Vec3f c = acc.sum.at<cv::Vec3f>(y, x) / w;
            out.at<cv::Vec3b>(y, x) =
                cv::Vec3b(cv::saturate_cast<uchar>(c[0]), cv::saturate_cast<uchar>(c[1]), cv::saturate_cast<uchar>(c[2]));
        }
    return out;
}

}  // namespace

cv::Rect2d warpedBounds(const std::vector<cv::Size>& sizes, const std::vector<cv::Matx33d>& toReference) {
    std::vector<cv::Point2d> all;
    for (size_t i = 0; i < sizes.size(); ++i)
        for (const auto& q : warpedOutline(sizes[i], toReference[i], 4)) all.push_back(q);
    return boundingBox(all);
}

Panorama stitch(const std::vector<cv::Mat>& images, const std::vector<cv::Matx33d>& toReference,
                const StitchParams& p, std::stop_token stop) {
    CV_Assert(images.size() == toReference.size() && !images.empty());
    // 2. The canvas: the warped bounds, shrunk if a wild homography would make it huge.
    std::vector<cv::Size> sizes;
    for (const auto& im : images) sizes.push_back(im.size());
    const cv::Rect2d b = warpedBounds(sizes, toReference);
    Panorama pano;
    pano.scale = std::min(1.0, std::sqrt(p.maxCanvasPixels / std::max(1.0, b.area())));
    const double s = pano.scale;
    const cv::Matx33d T(s, 0, -s * b.x, 0, s, -s * b.y, 0, 0, 1);  // reference frame -> canvas pixels
    const int W = std::max(1, int(std::ceil(b.width * s))), H = std::max(1, int(std::ceil(b.height * s)));

    // 3–4. Each image warped onto the canvas with its weights.
    Accumulator acc{cv::Mat(H, W, CV_32FC3, cv::Scalar::all(0)), cv::Mat(H, W, CV_32F, cv::Scalar(0))};
    for (size_t i = 0; i < images.size(); ++i) {
        checkStop(stop);
        const cv::Matx33d toCanvas = T * toReference[i];
        pano.toCanvas.push_back(toCanvas);
        pano.outlines.push_back(warpedOutline(images[i].size(), toCanvas));
        warpInto(acc, images[i], toCanvas, p.feather);
    }
    checkStop(stop);
    // 5. The weighted mean.
    pano.image = blend(acc);
    return pano;
}

}  // namespace vl::geometry
