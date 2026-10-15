#pragma once
// Making the flattened page look scanned: even white paper, dark ink, optionally pure black and white.
//
// A photo of paper is (reflectance) × (illumination): I(x) = R(x) · L(x). The paper's R is constant,
// the ink's is low; L varies slowly (lamp falloff, a shadow). Steps (cleanup.cpp):
//   1. estimateIllumination  L̂ = blur(close(I)) on a reduced page: the closing (max then min filter)
//                            replaces thin dark strokes by the paper around them, the blur smooths
//   2. flatten               F = I / (L̂^s · L̄^(1−s)): s = 1 divides the light out, s = 0 leaves it
//   3. contrastStretch       ink (1st percentile) → 0, paper (the median, as most of a page is paper) → 255
//   4. adaptiveThreshold     black where F < mean of the (2r+1)² window around the pixel − C, the window
//                            mean read from an integral image in O(1)
// Cost O(W·H) for each step whatever the window sizes (the closing works on a page 16 times smaller).

#include <opencv2/core.hpp>

namespace vl::scanner {

enum class ScanMode { Colour, Grey, BlackWhite };

struct CleanupParams {
    ScanMode mode = ScanMode::Colour;
    double shadowRemoval = 1.0;  // s in [0, 1]
    int window = 12;             // adaptive threshold: window radius r (pixels of the page)
    double C = 10;               // adaptive threshold: offset below the local mean (grey levels)
};

struct CleanupResult {
    cv::Mat illumination;  // 8-bit, the estimated L̂ (grey or BGR, like the output)
    cv::Mat scan;          // 8-bit: BGR in Colour mode, grey otherwise
};

// page: the rectified page, 8-bit BGR.
CleanupResult cleanUp(const cv::Mat& page, const CleanupParams& p);

// Step 1 on one channel (CV_32F): the slowly varying light, same size as the input.
cv::Mat estimateIllumination(const cv::Mat& I);

// The integral image S(x, y) = Σ_{x' < x, y' < y} f(x', y'), one row and one column larger than f (CV_64F).
cv::Mat integralImage(const cv::Mat& f);

// Step 4. grey: CV_32F or 8-bit. Returns CV_8U, 0 (ink) or 255 (paper).
cv::Mat adaptiveThreshold(const cv::Mat& grey, int r, double C);

}  // namespace vl::scanner
