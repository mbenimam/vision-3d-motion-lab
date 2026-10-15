// Canny's edge detector, written out step by step. Course "3D Vision and Motion": the scanner
// application (gradients and filtering are the tools of parts 3 and 11).
//
//   1. smoothedGradient       g = ∇(G_σ * I) with vl::gaussianBlur and vl::sobel
//   2. nonMaximumSuppression  thin ridges of ‖g‖ to one pixel, comparing along the gradient direction
//   3. automaticThresholds    T_high from the distribution of ridge strengths, T_low = ratio · T_high
//   4. hysteresis             flood fill from strong pixels through weak ones (8-connected)
//
// Why each step: blurring sets the scale (text strokes vanish, the page border stays); suppression
// gives thin edges, so a straight border votes for one line, not a band of lines; two thresholds
// keep a faint stretch of border that continues a strong one, and drop isolated faint noise.

#include "core/scanner/edges.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include "core/common/filters.hpp"

namespace vl::scanner {
namespace {

// Step 1: g = (∂/∂x, ∂/∂y)(G_σ * I) and ‖g‖ = √(gx² + gy²).
void smoothedGradient(const cv::Mat& grey, double sigma, EdgeMap& e) {
    const Gradient g = sobel(gaussianBlur(grey, sigma));
    e.dx = g.dx;
    e.dy = g.dy;
    e.magnitude.create(grey.size(), CV_32F);
    for (int y = 0; y < grey.rows; ++y)
        for (int x = 0; x < grey.cols; ++x) {
            const float gx = g.dx.at<float>(y, x), gy = g.dy.at<float>(y, x);
            e.magnitude.at<float>(y, x) = std::sqrt(gx * gx + gy * gy);
        }
}

// Step 2: a pixel survives if ‖g‖ is at least as large as at its two neighbours along the gradient.
// The gradient angle is rounded to one of four directions (0°, 45°, 90°, 135°), which names the
// neighbours to compare with: for a vertical edge (gradient along x) they are left and right.
cv::Mat nonMaximumSuppression(const EdgeMap& e) {
    cv::Mat ridge = cv::Mat::zeros(e.magnitude.size(), CV_32F);
    for (int y = 0; y < ridge.rows; ++y)
        for (int x = 0; x < ridge.cols; ++x) {
            const float m = e.magnitude.at<float>(y, x);
            if (m <= 0) continue;
            double angle = std::atan2(e.dy.at<float>(y, x), e.dx.at<float>(y, x)) * 180.0 / std::numbers::pi;
            if (angle < 0) angle += 180.0;  // a direction, not an orientation: θ and θ + 180° are the same
            int ox = 1, oy = 0;             // 0°: compare left and right
            if (angle >= 22.5 && angle < 67.5) {
                ox = 1;
                oy = 1;
            } else if (angle >= 67.5 && angle < 112.5) {
                ox = 0;
                oy = 1;
            } else if (angle >= 112.5 && angle < 157.5) {
                ox = -1;
                oy = 1;
            }
            const float before = vl::at(e.magnitude, y - oy, x - ox), after = vl::at(e.magnitude, y + oy, x + ox);
            // ≥ on one side and > on the other: a plateau two pixels wide keeps exactly one of them.
            if (m >= before && m > after) ridge.at<float>(y, x) = m;
        }
    return ridge;
}

// Step 3: most ridge pixels are faint (paper grain, desk texture, noise); the page border is among
// the strongest. T_high = the p-quantile of the ridge strengths adapts to the photo's contrast,
// where a fixed number would be too high for a pale desk and too low for a dark one.
double automaticHighThreshold(const cv::Mat& ridge, double percentile) {
    std::vector<float> values;
    for (int y = 0; y < ridge.rows; ++y)
        for (int x = 0; x < ridge.cols; ++x)
            if (ridge.at<float>(y, x) > 0) values.push_back(ridge.at<float>(y, x));
    if (values.empty()) return 1.0;
    const size_t k = std::min(values.size() - 1, size_t(percentile * double(values.size())));
    std::nth_element(values.begin(), values.begin() + std::ptrdiff_t(k), values.end());  // O(n), no full sort
    return std::max(1.0, double(values[k]));
}

// Step 4: start from every strong pixel and walk through 8-connected weak pixels with a stack.
// Each pixel is pushed at most once, so this is O(W·H).
cv::Mat hysteresis(const cv::Mat& ridge, double low, double high) {
    cv::Mat edges = cv::Mat::zeros(ridge.size(), CV_8U);
    std::vector<cv::Point> stack;
    for (int y = 0; y < ridge.rows; ++y)
        for (int x = 0; x < ridge.cols; ++x)
            if (ridge.at<float>(y, x) >= high) {
                edges.at<uchar>(y, x) = 255;
                stack.emplace_back(x, y);
            }
    while (!stack.empty()) {
        const cv::Point p = stack.back();
        stack.pop_back();
        for (int j = -1; j <= 1; ++j)
            for (int i = -1; i <= 1; ++i) {
                const int x = p.x + i, y = p.y + j;
                if (x < 0 || y < 0 || x >= ridge.cols || y >= ridge.rows) continue;
                if (edges.at<uchar>(y, x) == 0 && ridge.at<float>(y, x) >= low) {
                    edges.at<uchar>(y, x) = 255;
                    stack.emplace_back(x, y);
                }
            }
    }
    return edges;
}

}  // namespace

EdgeMap canny(const cv::Mat& grey, const CannyParams& p) {
    CV_Assert(grey.type() == CV_32F);
    EdgeMap e;
    smoothedGradient(grey, p.sigma, e);                                           // 1.
    const cv::Mat ridge = nonMaximumSuppression(e);                               // 2.
    e.high = p.automatic ? automaticHighThreshold(ridge, p.percentile) : p.high;  // 3.
    e.low = p.lowRatio * e.high;
    e.edges = hysteresis(ridge, e.low, e.high);  // 4.
    return e;
}

}  // namespace vl::scanner
