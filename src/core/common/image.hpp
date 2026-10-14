#pragma once
// Small image helpers shared by every area.

#include <opencv2/core.hpp>

namespace vl {

// Any 8-bit grey or BGR(A) image -> single-channel float in [0, 255].
cv::Mat toGrayF(const cv::Mat& img);

// Shrink so the longest side is at most maxSide (no-op when already small). Keeps the type.
cv::Mat limitSize(const cv::Mat& img, int maxSide, double* scale = nullptr);

// Map a float image to BGR with the turbo colormap over [lo, hi]; pixels where valid == 0 are black.
cv::Mat colorize(const cv::Mat& values, double lo, double hi, const cv::Mat& valid = cv::Mat());

// 8-bit grey/BGR -> 8-bit BGR, so every view can be shown the same way.
cv::Mat toBGR(const cv::Mat& img);

}  // namespace vl
