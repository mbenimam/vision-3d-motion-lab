#pragma once
// Measuring a flow against ground truth (Baker et al. 2011, "A database and evaluation methodology
// for optical flow"), and the Middlebury .flo file format.
//
//  - endpoint error   EPE = |w - w_gt|, in pixels: the distance between where we say the pixel
//    goes and where it really goes;
//  - angular error    AE = angle between (u, v, 1) and (u_gt, v_gt, 1), in degrees (Barron et al.):
//    the added 1 keeps it defined for zero motion and makes small errors on small motions count.
//
// Ground truth may hold "unknown" vectors (|u| or |v| > 1e9 in .flo files, or NaN): they are
// ignored. An estimate that is NaN where the truth is known is left out of the averages and
// counted in `missing`.

#include <string>

#include <opencv2/core.hpp>

namespace vl::flow {

bool isKnownFlow(const cv::Vec2f& w);  // finite and not a .flo "unknown" marker

struct FlowErrors {
    double epe = 0;        // mean endpoint error (px)
    double angular = 0;    // mean angular error (degrees)
    double over1 = 0;      // share of pixels with EPE > 1 px, in [0, 1]
    double over3 = 0;      // share of pixels with EPE > 3 px
    long count = 0;        // pixels evaluated (truth known and estimate finite)
    long missing = 0;      // truth known but no estimate
};

FlowErrors evaluateFlow(const cv::Mat& flow, const cv::Mat& truth);

// Per-pixel maps, CV_32F, NaN where truth is unknown or the estimate is missing.
cv::Mat endpointErrorMap(const cv::Mat& flow, const cv::Mat& truth);
cv::Mat angularErrorMap(const cv::Mat& flow, const cv::Mat& truth);  // degrees

// What the flow explains: |I1(x + w(x)) - I0(x)| on grey levels (CV_32F, bilinear, NaN where w is
// NaN or points outside frame 1). Works without ground truth.
cv::Mat brightnessResidual(const cv::Mat& frame0, const cv::Mat& frame1, const cv::Mat& flow);

// Middlebury .flo: "PIEH" (float 202021.25), int32 width, int32 height, then (u, v) float pairs in
// row-major order, little endian. readFlo throws std::runtime_error on a malformed file.
cv::Mat readFlo(const std::string& path);
void writeFlo(const std::string& path, const cv::Mat& flow);

}  // namespace vl::flow
