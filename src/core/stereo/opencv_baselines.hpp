#pragma once
// OpenCV's own stereo matchers, used as the baselines our implementations are compared to.
// Nothing here is part of the method being taught: these are thin wrappers that run OpenCV on the
// same pair and return its answer in our conventions (CV_32F disparity, NaN where it gives none).

#include <opencv2/core.hpp>

namespace vl::stereo::opencv {

struct Disparity {
    cv::Mat disparity;  // CV_32F, left coordinates, NaN where OpenCV found no match
    double ms = 0;      // wall-clock time of OpenCV's compute()
};

// cv::StereoBM: SAD block matching on prefiltered grey images (Konolige's algorithm).
// The window is (2·radius+1)², at least 5×5; the search range is rounded up to a multiple of 16.
Disparity blockMatching(const cv::Mat& left, const cv::Mat& right, int maxDisparity, int radius);

// cv::StereoSGBM: semi-global matching (Hirschmüller 2008), a smoothness term along 8 paths.
Disparity semiGlobal(const cv::Mat& left, const cv::Mat& right, int maxDisparity, int radius);

}  // namespace vl::stereo::opencv
