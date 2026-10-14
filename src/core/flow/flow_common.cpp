#include "core/flow/flow_common.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

#include "core/common/image.hpp"

namespace vl::flow {

namespace {

// d/dx and d/dy with the 5-tap kernel, borders replicated. O(W·H).
void gradient(const cv::Mat& I, cv::Mat& gx, cv::Mat& gy) {
    static float taps[5] = {1.f / 12, -8.f / 12, 0.f, 8.f / 12, -1.f / 12};
    static float unit[1] = {1.f};
    static const cv::Mat k(1, 5, CV_32F, taps), one(1, 1, CV_32F, unit);
    // sepFilter2D correlates: out(x) = (I(x-2) - 8 I(x-1) + 8 I(x+1) - I(x+2)) / 12.
    cv::sepFilter2D(I, gx, CV_32F, k, one, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::sepFilter2D(I, gy, CV_32F, one, k.t(), cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
}

}  // namespace

Derivatives derivatives(const cv::Mat& I0, const cv::Mat& I1, bool averageFrames) {
    CV_Assert(I0.type() == CV_32F && I1.type() == CV_32F && I0.size() == I1.size());
    Derivatives d;
    gradient(I0, d.Ix, d.Iy);
    if (averageFrames) {
        cv::Mat gx, gy;
        gradient(I1, gx, gy);
        d.Ix = 0.5f * (d.Ix + gx);
        d.Iy = 0.5f * (d.Iy + gy);
    }
    d.It = I1 - I0;
    return d;
}

std::vector<cv::Mat> gaussianPyramid(const cv::Mat& img, int levels, int minSide) {
    std::vector<cv::Mat> pyr{img};
    for (int l = 1; l < levels; ++l) {
        const cv::Mat& prev = pyr.back();
        if (std::min(prev.cols, prev.rows) / 2 < minSide) break;
        cv::Mat next;
        cv::pyrDown(prev, next);
        pyr.push_back(next);
    }
    return pyr;
}

cv::Mat warpImage(const cv::Mat& img, const cv::Mat& flow, cv::Mat* outside) {
    CV_Assert(flow.type() == CV_32FC2 && flow.size() == img.size());
    const int H = img.rows, W = img.cols;
    cv::Mat mapx(img.size(), CV_32F), mapy(img.size(), CV_32F);
    if (outside) outside->create(img.size(), CV_8U);
    for (int y = 0; y < H; ++y) {
        const auto* f = flow.ptr<cv::Vec2f>(y);
        float* mx = mapx.ptr<float>(y);
        float* my = mapy.ptr<float>(y);
        uchar* o = outside ? outside->ptr<uchar>(y) : nullptr;
        for (int x = 0; x < W; ++x) {
            // A NaN vector (no estimate) is treated as zero motion.
            const float u = std::isfinite(f[x][0]) ? f[x][0] : 0.f, v = std::isfinite(f[x][1]) ? f[x][1] : 0.f;
            mx[x] = float(x) + u;
            my[x] = float(y) + v;
            if (o) o[x] = (mx[x] < 0 || my[x] < 0 || mx[x] > float(W - 1) || my[x] > float(H - 1)) ? 255 : 0;
        }
    }
    cv::Mat out;
    cv::remap(img, out, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}

cv::Mat resizeFlow(const cv::Mat& flow, cv::Size size) {
    if (flow.size() == size) return flow.clone();
    cv::Mat out;
    cv::resize(flow, out, size, 0, 0, cv::INTER_LINEAR);
    const float sx = float(size.width) / float(flow.cols), sy = float(size.height) / float(flow.rows);
    out = out.mul(cv::Scalar(sx, sy));
    return out;
}

cv::Mat prepareFrame(const cv::Mat& img, double sigma) {
    cv::Mat g = vl::toGrayF(img);  // always a fresh buffer
    if (sigma > 0) cv::GaussianBlur(g, g, cv::Size(), sigma, sigma, cv::BORDER_REPLICATE);
    return g;
}

double meanMagnitude(const cv::Mat& flow) {
    double sum = 0;
    long n = 0;
    for (int y = 0; y < flow.rows; ++y) {
        const auto* f = flow.ptr<cv::Vec2f>(y);
        for (int x = 0; x < flow.cols; ++x) {
            if (!std::isfinite(f[x][0]) || !std::isfinite(f[x][1])) continue;
            sum += std::hypot(f[x][0], f[x][1]);
            ++n;
        }
    }
    return n ? sum / double(n) : 0.0;
}

}  // namespace vl::flow
