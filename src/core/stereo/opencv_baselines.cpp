#include "core/stereo/opencv_baselines.hpp"

#include <algorithm>
#include <limits>

#include "core/common/cv_compat.hpp"
#include "core/common/task.hpp"

namespace vl::stereo::opencv {
namespace {

int roundUpTo16(int d) { return std::max(16, (d + 16) / 16 * 16); }

// OpenCV returns 16·d in CV_16S, and (minDisparity - 1)·16 where it has no answer.
cv::Mat toFloat(const cv::Mat& fixedPoint) {
    cv::Mat d;
    fixedPoint.convertTo(d, CV_32F, 1.0 / 16.0);
    d.setTo(std::numeric_limits<float>::quiet_NaN(), fixedPoint < 0);
    return d;
}

cv::Mat grey(const cv::Mat& img) {
    if (img.channels() == 1) return img;
    cv::Mat g;
    cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    return g;
}

}  // namespace

Disparity blockMatching(const cv::Mat& left, const cv::Mat& right, int maxDisparity, int radius) {
    auto bm = cv::StereoBM::create(roundUpTo16(maxDisparity), std::max(5, 2 * radius + 1));
    cv::Mat raw;
    const Stopwatch sw;
    bm->compute(grey(left), grey(right), raw);
    return {toFloat(raw), sw.ms()};
}

Disparity semiGlobal(const cv::Mat& left, const cv::Mat& right, int maxDisparity, int radius) {
    const int block = std::max(3, 2 * radius + 1);
    const int channels = left.channels();
    const int range = roundUpTo16(maxDisparity);
    // The usual penalties: P1 for a 1-pixel disparity change, P2 for larger jumps.
    auto sgbm = cv::StereoSGBM::create(0, range, block, 8 * channels * block * block, 32 * channels * block * block, 1,
                                       0, 10, 100, 2, cv::StereoSGBM::MODE_SGBM_3WAY);
    // SGBM gives no answer in the leftmost `range` columns, where the full search range does not fit.
    // Our methods answer there with the shorter range that fits, so for a like-for-like comparison
    // both images get `range` replicated columns on the left, removed again from the result.
    cv::Mat paddedLeft, paddedRight, raw;
    cv::copyMakeBorder(left, paddedLeft, 0, 0, range, 0, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(right, paddedRight, 0, 0, range, 0, cv::BORDER_REPLICATE);
    const Stopwatch sw;
    sgbm->compute(paddedLeft, paddedRight, raw);
    const double ms = sw.ms();
    return {toFloat(raw.colRange(range, raw.cols)), ms};
}

}  // namespace vl::stereo::opencv
