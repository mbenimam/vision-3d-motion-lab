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
    // The usual penalties: P1 for a 1-pixel disparity change, P2 for larger jumps.
    auto sgbm = cv::StereoSGBM::create(0, roundUpTo16(maxDisparity), block, 8 * channels * block * block,
                                       32 * channels * block * block, 1, 0, 10, 100, 2,
                                       cv::StereoSGBM::MODE_SGBM_3WAY);
    cv::Mat raw;
    const Stopwatch sw;
    sgbm->compute(left, right, raw);
    return {toFloat(raw), sw.ms()};
}

}  // namespace vl::stereo::opencv
