#include "core/flow/flow_common.hpp"

#include <algorithm>
#include <cmath>

#include "core/common/filters.hpp"
#include "core/common/image.hpp"

namespace vl::flow {

namespace {

// The 5-tap derivative: weights of I(x − 2) ... I(x + 2).
constexpr float kDerivative[5] = {1.f / 12, -8.f / 12, 0.f, 8.f / 12, -1.f / 12};

// The derivative along the direction (dx, dy) = (1, 0) for ∂/∂x or (0, 1) for ∂/∂y, borders
// replicated. A 1D filter: no need for the full separable machinery. O(W·H).
cv::Mat differentiate(const cv::Mat& I, int dx, int dy) {
    cv::Mat out(I.size(), CV_32F);
    for (int y = 0; y < I.rows; ++y)
        for (int x = 0; x < I.cols; ++x) {
            float sum = 0;
            for (int k = -2; k <= 2; ++k) sum += kDerivative[k + 2] * vl::at(I, y + k * dy, x + k * dx);
            out.at<float>(y, x) = sum;
        }
    return out;
}

Derivatives spatialDerivatives(const cv::Mat& I) { return {differentiate(I, 1, 0), differentiate(I, 0, 1), {}}; }

}  // namespace

Derivatives derivatives(const cv::Mat& I0, const cv::Mat& I1, bool averageFrames) {
    CV_Assert(I0.type() == CV_32F && I1.type() == CV_32F && I0.size() == I1.size());
    Derivatives d = spatialDerivatives(I0);
    if (averageFrames) {
        const Derivatives d1 = spatialDerivatives(I1);
        d.Ix = 0.5f * (d.Ix + d1.Ix);
        d.Iy = 0.5f * (d.Iy + d1.Iy);
    }
    d.It = I1 - I0;
    return d;
}

Derivatives linearise(const cv::Mat& I0, const cv::Mat& I1, const cv::Mat& w) {
    cv::Mat outside;
    const cv::Mat I1w = warpImage(I1, w, &outside);
    Derivatives d = derivatives(I0, I1w, true);
    d.Ix.setTo(0, outside);
    d.Iy.setTo(0, outside);
    d.It.setTo(0, outside);
    return d;
}

std::vector<cv::Mat> buildPyramid(const cv::Mat& img, int levels, int minSide) {
    std::vector<cv::Mat> pyramid{img};
    for (int l = 1; l < levels; ++l) {
        const cv::Mat& finer = pyramid.back();
        if (std::min(finer.cols, finer.rows) / 2 < minSide) break;
        pyramid.push_back(vl::pyramidDown(finer));
    }
    return pyramid;
}

cv::Mat warpImage(const cv::Mat& img, const cv::Mat& flow, cv::Mat* outside) {
    CV_Assert(img.type() == CV_32F && flow.type() == CV_32FC2 && flow.size() == img.size());
    const float maxX = float(img.cols - 1), maxY = float(img.rows - 1);
    cv::Mat out(img.size(), CV_32F);
    if (outside) outside->create(img.size(), CV_8U);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
            const cv::Vec2f w = flow.at<cv::Vec2f>(y, x);
            const float u = std::isfinite(w[0]) ? w[0] : 0.f, v = std::isfinite(w[1]) ? w[1] : 0.f;
            const float xs = float(x) + u, ys = float(y) + v;  // where pixel (x, y) of frame 0 went
            out.at<float>(y, x) = vl::bilinear(img, xs, ys);
            if (outside) outside->at<uchar>(y, x) = (xs < 0 || ys < 0 || xs > maxX || ys > maxY) ? 255 : 0;
        }
    return out;
}

cv::Mat upsampleFlow(const cv::Mat& flow, cv::Size size, float factor) {
    CV_Assert(flow.type() == CV_32FC2 && factor > 0);
    std::vector<cv::Mat> uv;
    cv::split(flow, uv);  // bilinear works on one channel at a time
    cv::Mat out(size, CV_32FC2);
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x) {
            // Pyramid level l keeps the pixels 0, 2, 4... of level l − 1, so fine x is coarse x / 2.
            const float xc = float(x) / factor, yc = float(y) / factor;
            out.at<cv::Vec2f>(y, x) = {factor * vl::bilinear(uv[0], xc, yc), factor * vl::bilinear(uv[1], xc, yc)};
        }
    return out;
}

cv::Mat prepareFrame(const cv::Mat& img, double sigma) {
    const cv::Mat g = vl::toGrayF(img);
    return sigma > 0 ? vl::gaussianBlur(g, sigma) : g.clone();
}

double meanMagnitude(const cv::Mat& flow) {
    double sum = 0;
    long n = 0;
    for (int y = 0; y < flow.rows; ++y)
        for (int x = 0; x < flow.cols; ++x) {
            const cv::Vec2f w = flow.at<cv::Vec2f>(y, x);
            if (!std::isfinite(w[0]) || !std::isfinite(w[1])) continue;
            sum += std::hypot(w[0], w[1]);
            ++n;
        }
    return n ? sum / double(n) : 0.0;
}

}  // namespace vl::flow
