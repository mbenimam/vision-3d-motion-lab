#pragma once
// A document scanner from scratch: photo of a page in, flat clean page out.
// Course "3D Vision and Motion": an application of part 2 (homographies, the DLT).
//
// The pipeline, one file per stage:
//   1. findPage        (this file)  reduce the photo, Canny edges (edges.hpp), Hough lines (hough.hpp),
//                                   the best quadrilateral (quad.hpp), its sides refined at full size
//   2. orderCorners    (quad.hpp)   top-left, top-right, bottom-right, bottom-left
//   3. rectify         (rectify.hpp) H from the corners to a W × H rectangle by the DLT, inverse warp
//   4. cleanUp         (cleanup.hpp) divide the light out, stretch, optionally an adaptive threshold
//
// The user can replace step 1 by dragging the corners: scanDocument then takes them as given.

#include <optional>
#include <stop_token>

#include <opencv2/core.hpp>

#include "core/scanner/cleanup.hpp"
#include "core/scanner/edges.hpp"
#include "core/scanner/hough.hpp"
#include "core/scanner/quad.hpp"
#include "core/scanner/rectify.hpp"

namespace vl::scanner {

struct ScanParams {
    int detectionSide = 640;  // the photo is halved until its longer side is at most this (pixels)
    CannyParams canny;
    HoughParams hough;
    QuadParams quad;
    bool refineSides = true;  // re-fit the sides on the full-size photo, to a fraction of a pixel
    PageRatio ratio = PageRatio::Auto;
    CleanupParams cleanup;
};

struct PageDetection {
    bool found = false;
    Quad corners;                       // full-size photo pixels, ordered; a default inset when nothing was found
    double scale = 1;                   // full size / detection size (a power of two)
    EdgeMap edges;                      // at the detection size
    HoughResult hough;                  // at the detection size
    std::optional<QuadCandidate> quad;  // the winner, at the detection size
};

// Step 1. photo: 8-bit grey or BGR.
PageDetection findPage(const cv::Mat& photo, const ScanParams& p, std::stop_token stop = {});

struct ScanResult {
    PageDetection detection;  // empty when the corners were given
    Quad corners;             // the corners used
    bool manual = false;      // true when they were given
    cv::Size size;            // of the flat page
    cv::Matx33d H;            // photo → flat page
    cv::Mat rectified;        // 8-bit BGR flat page, before clean-up
    CleanupResult clean;
};

// Steps 1-4. photo: 8-bit BGR. `given`: corners chosen by the user (any order), used instead of step 1.
ScanResult scanDocument(const cv::Mat& photo, const ScanParams& p, const std::optional<Quad>& given = std::nullopt,
                        std::stop_token stop = {});

// Mean distance between corresponding corners of two quadrilaterals (pixels).
double cornerError(const Quad& a, const Quad& b);

// Peak signal-to-noise ratio of two 8-bit images of the same size, compared in grey:
// PSNR = 10 log₁₀(255² / mean (a − b)²), in dB (higher is closer; 99 for identical images).
double psnr(const cv::Mat& a, const cv::Mat& b);

}  // namespace vl::scanner
