#pragma once
// Canny's edge detector, the first step of the document scanner (see scanner.hpp).
//
// An edge is a place where the intensity changes fast *across* a line and slowly along it. Canny
// (1986) turns that into four steps (edges.cpp, one function each):
//   1. smoothedGradient     g = ∇(G_σ * I): blur, then Sobel; ‖g‖ is the edge strength, its angle the normal
//   2. nonMaximumSuppression keep a pixel only where ‖g‖ is a maximum along the normal: edges one pixel thin
//   3. automaticThresholds   T_high = a high percentile of ‖g‖ on the thin edges, T_low = ratio · T_high
//   4. hysteresis            keep strong pixels (‖g‖ ≥ T_high) and the weak ones (≥ T_low) connected to them
// Cost O(W·H·r) for the blur of radius r, O(W·H) for the rest.

#include <opencv2/core.hpp>

namespace vl::scanner {

struct CannyParams {
    double sigma = 2.0;        // blur before differentiating (pixels): smooths noise and small print away
    bool automatic = true;     // thresholds from the image's own gradient statistics (step 3)
    double percentile = 0.92;  // automatic: T_high = this quantile of the thinned gradient magnitudes
    double high = 6.0;         // manual T_high, grey levels per pixel
    double lowRatio = 0.4;     // T_low = lowRatio · T_high
};

struct EdgeMap {
    cv::Mat edges;             // CV_8U, 255 on an edge pixel
    cv::Mat magnitude;         // CV_32F ‖g‖, grey levels per pixel
    cv::Mat dx, dy;            // CV_32F gradient components
    double low = 0, high = 0;  // the thresholds that were used
};

// grey: CV_32F image in [0, 255].
EdgeMap canny(const cv::Mat& grey, const CannyParams& p);

}  // namespace vl::scanner
