#include "core/geometry/stitch.hpp"

#include <algorithm>
#include <cmath>

#include "core/common/image.hpp"
#include "core/common/task.hpp"
#include "core/geometry/homography.hpp"

namespace vl::geometry {

std::vector<cv::Matx33d> chainToReference(const std::vector<cv::Matx33d>& links, int reference) {
    const int n = static_cast<int>(links.size()) + 1;
    CV_Assert(reference >= 0 && reference < n);
    std::vector<cv::Matx33d> out(size_t(n), cv::Matx33d::eye());
    // Left of the reference: image i -> i+1 -> ... -> r.
    for (int i = reference - 1; i >= 0; --i) out[size_t(i)] = out[size_t(i + 1)] * links[size_t(i)];
    // Right of it: image i -> i-1 -> ... -> r, each link inverted.
    for (int i = reference + 1; i < n; ++i) out[size_t(i)] = out[size_t(i - 1)] * links[size_t(i - 1)].inv();
    for (auto& H : out)
        if (std::abs(H(2, 2)) > 1e-12) H *= 1.0 / H(2, 2);
    return out;
}

std::vector<cv::Point2d> warpedOutline(cv::Size size, const cv::Matx33d& H, int perEdge) {
    const double W = size.width, Hh = size.height;
    const cv::Point2d c[] = {{0, 0}, {W, 0}, {W, Hh}, {0, Hh}};
    std::vector<cv::Point2d> out;
    out.reserve(size_t(4 * perEdge));
    for (int e = 0; e < 4; ++e)
        for (int k = 0; k < perEdge; ++k) {
            const double t = double(k) / perEdge;
            out.push_back(transform(H, c[e] + (c[(e + 1) % 4] - c[e]) * t));
        }
    return out;
}

cv::Rect2d warpedBounds(const std::vector<cv::Size>& sizes, const std::vector<cv::Matx33d>& toReference) {
    double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
    for (size_t i = 0; i < sizes.size(); ++i)
        for (const auto& q : warpedOutline(sizes[i], toReference[i], 4)) {
            if (!std::isfinite(q.x) || !std::isfinite(q.y)) continue;
            x0 = std::min(x0, q.x), y0 = std::min(y0, q.y), x1 = std::max(x1, q.x), y1 = std::max(y1, q.y);
        }
    if (x1 < x0) return {};
    return {x0, y0, x1 - x0, y1 - y0};
}

Panorama stitch(const std::vector<cv::Mat>& images, const std::vector<cv::Matx33d>& toReference,
                const StitchParams& p, std::stop_token stop) {
    CV_Assert(images.size() == toReference.size() && !images.empty());
    std::vector<cv::Size> sizes;
    for (const auto& im : images) sizes.push_back(im.size());
    const cv::Rect2d b = warpedBounds(sizes, toReference);

    Panorama pano;
    pano.scale = std::min(1.0, std::sqrt(p.maxCanvasPixels / std::max(1.0, b.area())));
    const double s = pano.scale;
    const cv::Matx33d T(s, 0, -s * b.x, 0, s, -s * b.y, 0, 0, 1);  // reference frame -> canvas pixels
    const int W = std::max(1, int(std::ceil(b.width * s))), H = std::max(1, int(std::ceil(b.height * s)));

    cv::Mat acc(H, W, CV_32FC3, cv::Scalar::all(0)), wsum(H, W, CV_32F, cv::Scalar(0));
    for (size_t i = 0; i < images.size(); ++i) {
        checkStop(stop);
        const cv::Mat src = toBGR(images[i]);
        const cv::Matx33d toCanvas = T * toReference[i];
        const cv::Matx33d inv = toCanvas.inv();  // canvas -> image i
        pano.toCanvas.push_back(toCanvas);
        pano.outlines.push_back(warpedOutline(src.size(), toCanvas));

        // Only rows/columns inside the image's warped box are visited.
        cv::Rect2d box(0, 0, 0, 0);
        {
            double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
            for (const auto& q : pano.outlines.back())
                x0 = std::min(x0, q.x), y0 = std::min(y0, q.y), x1 = std::max(x1, q.x), y1 = std::max(y1, q.y);
            box = cv::Rect2d(x0, y0, x1 - x0, y1 - y0);
        }
        const int ys = std::max(0, int(std::floor(box.y))), ye = std::min(H, int(std::ceil(box.y + box.height)) + 1);
        const int xs = std::max(0, int(std::floor(box.x))), xe = std::min(W, int(std::ceil(box.x + box.width)) + 1);
        const int sw = src.cols, sh = src.rows;
        const float half = 0.5f * float(std::min(sw, sh));
        const bool feather = p.feather;

        cv::parallel_for_(cv::Range(ys, std::max(ys, ye)), [&](const cv::Range& rows) {
            for (int y = rows.start; y < rows.end; ++y) {
                cv::Vec3f* a = acc.ptr<cv::Vec3f>(y);
                float* ws = wsum.ptr<float>(y);
                // Homogeneous source point of (xs, y); each step in x adds the first column of inv.
                double hx = inv(0, 0) * xs + inv(0, 1) * y + inv(0, 2);
                double hy = inv(1, 0) * xs + inv(1, 1) * y + inv(1, 2);
                double hw = inv(2, 0) * xs + inv(2, 1) * y + inv(2, 2);
                for (int x = xs; x < xe; ++x, hx += inv(0, 0), hy += inv(1, 0), hw += inv(2, 0)) {
                    if (hw <= 0) continue;  // behind the camera
                    const double u = hx / hw, v = hy / hw;
                    if (u < 0 || v < 0 || u > sw - 1 || v > sh - 1) continue;
                    const int u0 = std::min(int(u), sw - 2), v0 = std::min(int(v), sh - 2);
                    const float fu = float(u - u0), fv = float(v - v0);
                    const cv::Vec3b* r0 = src.ptr<cv::Vec3b>(v0);
                    const cv::Vec3b* r1 = src.ptr<cv::Vec3b>(v0 + 1);
                    cv::Vec3f c;
                    for (int k = 0; k < 3; ++k)
                        c[k] = (1 - fv) * ((1 - fu) * r0[u0][k] + fu * r0[u0 + 1][k]) +
                               fv * ((1 - fu) * r1[u0][k] + fu * r1[u0 + 1][k]);
                    if (feather) {
                        // Distance to the nearest border of image i, normalised; +1 keeps edges non-zero.
                        const float d = float(std::min({u, v, sw - 1 - u, sh - 1 - v})) + 1.f;
                        const float w = std::min(d / half, 1.f);
                        a[x] += w * c;
                        ws[x] += w;
                    } else {
                        a[x] = c;
                        ws[x] = 1.f;
                    }
                }
            }
        });
    }
    checkStop(stop);

    pano.image.create(H, W, CV_8UC3);
    for (int y = 0; y < H; ++y) {
        const cv::Vec3f* a = acc.ptr<cv::Vec3f>(y);
        const float* ws = wsum.ptr<float>(y);
        cv::Vec3b* o = pano.image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; ++x)
            o[x] = ws[x] > 0 ? cv::Vec3b(cv::saturate_cast<uchar>(a[x][0] / ws[x]), cv::saturate_cast<uchar>(a[x][1] / ws[x]),
                                         cv::saturate_cast<uchar>(a[x][2] / ws[x]))
                             : cv::Vec3b(0, 0, 0);
    }
    return pano;
}

}  // namespace vl::geometry
