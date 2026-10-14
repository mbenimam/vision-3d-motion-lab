// The document scanner's outline: each stage is a call into its own file (see scanner.hpp).
//
//   findPage      1. reduce the photo with a Gaussian pyramid (detection at ~640 px is fast and
//                    blurs small print away)
//                 2. Canny edges                     3. Hough lines
//                 4. best quadrilateral              5. refine its sides on the full photo,
//                                                       corners = intersections of consecutive sides
//   scanDocument  findPage (or the user's corners) → order → output size → rectify → clean up

#include "core/scanner/scanner.hpp"

#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "core/common/filters.hpp"
#include "core/common/image.hpp"
#include "core/common/task.hpp"

namespace vl::scanner {
namespace {

// Step 5: each side re-fitted on the full-size grey photo, and the corners recomputed as the
// intersections of consecutive sides. A side that moved implausibly far keeps its coarse corners.
Quad refineCorners(const cv::Mat& grey, const Quad& coarse) {
    std::array<HoughLine, 4> sides;
    for (size_t i = 0; i < 4; ++i) sides[i] = refineSide(grey, coarse[i], coarse[(i + 1) % 4]);
    Quad refined;
    for (size_t i = 0; i < 4; ++i) {
        // Corner i is where side i − 1 (from corner i − 1) meets side i (to corner i + 1).
        cv::Point2d p;
        const bool ok = intersect(sides[(i + 3) % 4], sides[i], p);
        refined[i] = ok && std::hypot(p.x - coarse[i].x, p.y - coarse[i].y) < 12.0 ? p : coarse[i];
    }
    return refined;
}

// When nothing looks like a page: a rectangle 5 % inside the photo, for the user to drag.
Quad insetCorners(cv::Size s) {
    const double mx = 0.05 * s.width, my = 0.05 * s.height;
    return {cv::Point2d(mx, my), cv::Point2d(s.width - 1 - mx, my), cv::Point2d(s.width - 1 - mx, s.height - 1 - my),
            cv::Point2d(mx, s.height - 1 - my)};
}

}  // namespace

PageDetection findPage(const cv::Mat& photo, const ScanParams& p, std::stop_token stop) {
    PageDetection d;
    const cv::Mat grey = toGrayF(photo);
    // 1. Halve until small enough. Pyramid pixel (x, y) is full-size pixel (2x, 2y): exact scaling.
    cv::Mat small = grey;
    while (std::max(small.cols, small.rows) > p.detectionSide) {
        small = pyramidDown(small);
        d.scale *= 2;
    }
    checkStop(stop);
    d.edges = canny(small, p.canny);  // 2.
    checkStop(stop);
    d.hough = houghLines(d.edges.edges, d.edges.dx, d.edges.dy, p.hough);  // 3.
    checkStop(stop);
    d.quad = bestQuadrilateral(d.hough.lines, d.edges, p.quad);  // 4.
    d.found = d.quad.has_value();
    if (!d.found) {
        d.corners = insetCorners(photo.size());
        return d;
    }
    for (size_t i = 0; i < 4; ++i) d.corners[i] = d.quad->corners[i] * d.scale;
    if (p.refineSides) d.corners = refineCorners(grey, d.corners);  // 5.
    return d;
}

ScanResult scanDocument(const cv::Mat& photo, const ScanParams& p, const std::optional<Quad>& given,
                        std::stop_token stop) {
    ScanResult r;
    const cv::Mat bgr = toBGR(photo);
    if (given) {
        r.manual = true;
        r.corners = orderCorners({given->begin(), given->end()});
    } else {
        r.detection = findPage(bgr, p, stop);
        r.corners = r.detection.corners;
    }
    checkStop(stop);
    r.size = outputSize(r.corners, p.ratio);
    r.H = pageHomography(r.corners, r.size);
    r.rectified = warpInverse(bgr, r.H, r.size);
    checkStop(stop);
    r.clean = cleanUp(r.rectified, p.cleanup);
    return r;
}

double cornerError(const Quad& a, const Quad& b) {
    double sum = 0;
    for (size_t i = 0; i < 4; ++i) sum += std::hypot(a[i].x - b[i].x, a[i].y - b[i].y);
    return sum / 4;
}

double psnr(const cv::Mat& a, const cv::Mat& b) {
    CV_Assert(a.size() == b.size());
    const cv::Mat ga = toGrayF(a), gb = toGrayF(b);
    double sse = 0;
    for (int y = 0; y < ga.rows; ++y)
        for (int x = 0; x < ga.cols; ++x) {
            const double e = ga.at<float>(y, x) - gb.at<float>(y, x);
            sse += e * e;
        }
    const double mse = sse / double(ga.total());
    return mse < 1e-10 ? 99.0 : 10 * std::log10(255.0 * 255.0 / mse);
}

}  // namespace vl::scanner
