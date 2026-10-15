#pragma once
// OpenCV's own optical-flow methods, the baselines our implementations are compared to. Nothing
// here is part of the methods being taught: thin wrappers that run OpenCV on the same pair and
// return its answer in our conventions (CV_32FC2 flow, frame 0 coordinates, NaN where it has none).

#include <opencv2/core.hpp>

namespace vl::flow::opencv {

struct Flow {
    cv::Mat flow;   // CV_32FC2
    double ms = 0;  // wall-clock time of OpenCV's call, grey conversion excluded
};

// cv::calcOpticalFlowPyrLK (Bouguet's sparse pyramidal LK) run densely: one point per pixel, every
// pixel. The square window is (2·windowRadius+1)², `levels` pyramid levels (maxLevel = levels − 1),
// OpenCV's default stopping rule (30 iterations or a step under 0.01 px). NaN where its status is 0.
Flow pyramidalLucasKanade(const cv::Mat& frame0, const cv::Mat& frame1, int windowRadius, int levels);

// cv::calcOpticalFlowFarneback (polynomial expansion, Farnebäck 2003): the classic dense method of
// OpenCV, with its usual settings (pyramid scale 0.5, 15 px window, 3 iterations, poly_n 5, σ 1.2).
Flow farneback(const cv::Mat& frame0, const cv::Mat& frame1, int levels);

// cv::DISOpticalFlow (Dense Inverse Search, Kroeger et al. 2016), PRESET_MEDIUM: patch matching by
// inverse-compositional search, densified and refined variationally.
Flow dis(const cv::Mat& frame0, const cv::Mat& frame1);

}  // namespace vl::flow::opencv
