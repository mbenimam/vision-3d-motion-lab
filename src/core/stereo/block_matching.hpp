#pragma once
// Local stereo: for each pixel of the left image, slide a window along the same row of the
// right image and keep the shift (disparity) where the windows look most alike.
//
// Cost per pixel and disparity is computed once for the whole image, then summed over the window
// with an integral image, so the price is O(W·H·D) whatever the window size.

#include <stop_token>
#include <vector>

#include <opencv2/core.hpp>

#include "core/common/task.hpp"

namespace vl::stereo {

enum class Cost { SAD, SSD, ZNCC };

struct BlockMatchParams {
    int minDisparity = 0;
    int maxDisparity = 48;     // inclusive
    int radius = 3;            // window is (2r+1)²
    Cost cost = Cost::ZNCC;
    bool leftRightCheck = true;
    float lrTolerance = 1.0f;  // |dL(x) - dR(x - dL)| allowed, in pixels
    bool subpixel = true;      // parabola through the costs at d-1, d, d+1
    bool keepVolume = false;   // keep the cost volume (for per-pixel cost curves)
};

struct BlockMatchResult {
    cv::Mat disparity;          // CV_32F, left coordinates; NaN where rejected
    cv::Mat rightDisparity;     // CV_32F, right coordinates (winner of the right-to-left search)
    cv::Mat rejected;           // CV_8U, 255 where the left-right check failed
    std::vector<cv::Mat> volume;  // optional: one CV_32F cost image per disparity (lower = better)
    int minDisparity = 0;
};

// left, right: rectified 8-bit grey or BGR images of the same size.
BlockMatchResult blockMatch(const cv::Mat& left, const cv::Mat& right, const BlockMatchParams& p,
                            std::stop_token stop = {}, const ProgressFn& progress = {});

// Sum of f over the (2r+1)² window centred on each pixel (borders: window clipped to the image),
// computed from an integral image in O(1) per pixel.
cv::Mat boxSum(const cv::Mat& f32, int r);

// Share of pixels with |d - truth| > threshold, among pixels where mask != 0 and truth is valid
// (> 0). Rejected (NaN) estimates count as errors. Returns a fraction in [0, 1].
double badPixelRate(const cv::Mat& disparity, const cv::Mat& truth, const cv::Mat& mask, double threshold);

}  // namespace vl::stereo
