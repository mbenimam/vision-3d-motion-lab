#include "core/common/filters.hpp"

#include <algorithm>
#include <cmath>

namespace vl {

float bilinear(const cv::Mat& f, float x, float y) {
    const int x0 = int(std::floor(x)), y0 = int(std::floor(y));
    const float ax = x - float(x0), ay = y - float(y0);  // position inside the cell, in [0, 1)
    const float top = (1 - ax) * at(f, y0, x0) + ax * at(f, y0, x0 + 1);
    const float bottom = (1 - ax) * at(f, y0 + 1, x0) + ax * at(f, y0 + 1, x0 + 1);
    return (1 - ay) * top + ay * bottom;
}

cv::Mat convolve(const cv::Mat& f, const cv::Mat& kernel) {
    CV_Assert(f.type() == CV_32F && kernel.type() == CV_32F && kernel.rows % 2 == 1 && kernel.cols % 2 == 1);
    const int ry = kernel.rows / 2, rx = kernel.cols / 2;
    cv::Mat g(f.size(), CV_32F);
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x) {
            float sum = 0;
            for (int j = -ry; j <= ry; ++j)
                for (int i = -rx; i <= rx; ++i) sum += kernel.at<float>(j + ry, i + rx) * at(f, y + j, x + i);
            g.at<float>(y, x) = sum;
        }
    return g;
}

cv::Mat convolveSeparable(const cv::Mat& f, const std::vector<float>& horizontal, const std::vector<float>& vertical) {
    CV_Assert(f.type() == CV_32F && horizontal.size() % 2 == 1 && vertical.size() % 2 == 1);
    const int rx = int(horizontal.size()) / 2, ry = int(vertical.size()) / 2;
    // Pass 1, along rows: h(x, y) = Σ_i a(i) f(x + i, y)
    cv::Mat h(f.size(), CV_32F);
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x) {
            float sum = 0;
            for (int i = -rx; i <= rx; ++i) sum += horizontal[size_t(i + rx)] * at(f, y, x + i);
            h.at<float>(y, x) = sum;
        }
    // Pass 2, along columns: g(x, y) = Σ_j b(j) h(x, y + j)
    cv::Mat g(f.size(), CV_32F);
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x) {
            float sum = 0;
            for (int j = -ry; j <= ry; ++j) sum += vertical[size_t(j + ry)] * at(h, y + j, x);
            g.at<float>(y, x) = sum;
        }
    return g;
}

std::vector<float> gaussianKernel(double sigma) {
    CV_Assert(sigma > 0);
    const int r = std::max(1, int(std::ceil(3 * sigma)));
    std::vector<float> k(size_t(2 * r + 1));
    double total = 0;
    for (int t = -r; t <= r; ++t) {
        const double v = std::exp(-double(t * t) / (2 * sigma * sigma));
        k[size_t(t + r)] = float(v);
        total += v;
    }
    for (float& v : k) v = float(v / total);  // weights sum to 1: a flat image stays flat
    return k;
}

cv::Mat gaussianBlur(const cv::Mat& f, double sigma) {
    const auto g = gaussianKernel(sigma);
    return convolveSeparable(f, g, g);
}

Gradient sobel(const cv::Mat& f) {
    // [−1 0 1] differentiates, [1 2 1] smooths across. Divided by 2 and by 4: a slope of 1 gives 1.
    const std::vector<float> diff = {-0.5f, 0.f, 0.5f}, smooth = {0.25f, 0.5f, 0.25f};
    return {convolveSeparable(f, diff, smooth), convolveSeparable(f, smooth, diff)};
}

namespace {

// Offsets of the pixels of a disc of radius r (the structuring element).
std::vector<cv::Point> disc(int r) {
    std::vector<cv::Point> offsets;
    for (int j = -r; j <= r; ++j)
        for (int i = -r; i <= r; ++i)
            if (i * i + j * j <= r * r) offsets.emplace_back(i, j);
    return offsets;
}

// Erosion keeps a pixel only if the whole disc around it is set; dilation sets it if any is.
cv::Mat morph(const cv::Mat& mask, int r, bool isErosion) {
    CV_Assert(mask.type() == CV_8U);
    const auto offsets = disc(r);
    cv::Mat out(mask.size(), CV_8U);
    for (int y = 0; y < mask.rows; ++y)
        for (int x = 0; x < mask.cols; ++x) {
            bool all = true, any = false;
            for (const auto& o : offsets) {
                const int yy = std::clamp(y + o.y, 0, mask.rows - 1), xx = std::clamp(x + o.x, 0, mask.cols - 1);
                const bool on = mask.at<uchar>(yy, xx) != 0;
                all = all && on;
                any = any || on;
            }
            out.at<uchar>(y, x) = (isErosion ? all : any) ? 255 : 0;
        }
    return out;
}

}  // namespace

cv::Mat erode(const cv::Mat& mask, int r) { return morph(mask, r, true); }
cv::Mat dilate(const cv::Mat& mask, int r) { return morph(mask, r, false); }
cv::Mat open(const cv::Mat& mask, int r) { return dilate(erode(mask, r), r); }
cv::Mat close(const cv::Mat& mask, int r) { return erode(dilate(mask, r), r); }

cv::Mat pyramidDown(const cv::Mat& f) {
    // Blur first: keeping every other pixel halves the sampling rate, so detail finer than the new
    // grid must be removed or it would come back as aliasing.
    const cv::Mat b = gaussianBlur(f, 1.0);
    cv::Mat out((f.rows + 1) / 2, (f.cols + 1) / 2, CV_32F);
    for (int y = 0; y < out.rows; ++y)
        for (int x = 0; x < out.cols; ++x) out.at<float>(y, x) = b.at<float>(2 * y, 2 * x);
    return out;
}

std::vector<cv::Mat> gaussianPyramid(const cv::Mat& f, int levels) {
    std::vector<cv::Mat> p{f};
    for (int l = 1; l < levels; ++l) p.push_back(pyramidDown(p.back()));
    return p;
}

}  // namespace vl
