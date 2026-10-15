#pragma once
// The classic OpenCV document-scanner recipe, the baseline our scanner is compared to. Nothing here
// is part of the method being taught: OpenCV's own functions, in the order every tutorial uses them.
//
//   cv::resize → cv::GaussianBlur → cv::Canny → cv::findContours → cv::approxPolyDP (the largest
//   convex contour with 4 vertices) → cv::getPerspectiveTransform → cv::warpPerspective →
//   cv::adaptiveThreshold (black and white) or a dilate / medianBlur background division (grey, colour)

#include <opencv2/core.hpp>

#include "core/scanner/cleanup.hpp"
#include "core/scanner/quad.hpp"

namespace vl::scanner::opencv {

struct Scan {
    bool found = false;  // a 4-vertex contour was found; otherwise the corners are the whole photo
    Quad corners;        // full-size photo pixels, ordered
    cv::Mat rectified;   // 8-bit BGR
    cv::Mat scan;        // 8-bit, BGR in Colour mode, grey otherwise
    double ms = 0;       // the whole recipe
};

// photo: 8-bit BGR. The same detection size, Canny thresholds (in our units: grey levels per pixel of
// the 1/8-normalised Sobel) and output size as ours, so the comparison is about the method.
Scan documentScan(const cv::Mat& photo, int detectionSide, double cannyLow, double cannyHigh, cv::Size outSize,
                  const CleanupParams& cleanup);

// The flat page from given corners (cv::getPerspectiveTransform + cv::warpPerspective, bilinear).
cv::Mat rectify(const cv::Mat& photo, const Quad& corners, cv::Size size);

}  // namespace vl::scanner::opencv
