#pragma once
// Pictures and numbers shared by the seed-propagation and graph-cut topics: the working pair, the
// disparity and error views, the comparison with an OpenCV matcher, and the click that shows a
// pixel's match in the right image.

#include <opencv2/core.hpp>

#include "app/module.hpp"

namespace vl::app::stereo_views {

// The pair the method works on, optionally shrunk by `factor` (truth is divided by it as well).
struct Working {
    cv::Mat left, right;   // 8-bit BGR
    cv::Mat truth;         // CV_32F, 0 = unknown; empty without ground truth
    cv::Mat visible;       // CV_8U, 255 where the left pixel is seen in the right image; empty without truth
    int maxDisparity = 0;  // search range at this resolution
    int factor = 1;
};
Working prepare(const Inputs& in, int factor);

// Turbo-coloured disparity, black where d is NaN, with a probe that also reads the truth.
View disparityView(const QString& name, const cv::Mat& disparity, const Working& w, const QString& nanText);

// Bad-pixel share (|d − truth| > 1 px, NaN counts as wrong) on the visible pixels with known truth.
double badRate(const cv::Mat& disparity, const Working& w);

// The same, counted only where the method gave an answer (for methods that leave holes on purpose).
double answeredBadRate(const cv::Mat& disparity, const Working& w);

// Percentage of pixels with a disparity (not NaN).
double answeredPercent(const cv::Mat& disparity);

// Percentage of the truly occluded pixels (known truth, hidden in the right image) left without an
// answer: how well a method finds occlusions. 0 without ground truth.
double occlusionsLeftOpenPercent(const cv::Mat& disparity, const Working& w);

// The left image greyed out; on visible pixels with known truth: wrong disparities in pink, NaN in blue.
cv::Mat errorImage(const cv::Mat& disparity, const Working& w);

// Left image at half brightness, in grey, ready for colour overlays.
cv::Mat dimmed(const cv::Mat& bgr);

// "Compare with OpenCV" for a stereo topic: cv::StereoSGBM on the same working pair, its disparity
// as the view "OpenCV · Disparity", and the rows bad pixels, (optionally) errors among answered
// pixels, pixels with an answer, (optionally) true occlusions left unmatched, and run time.
struct OursForComparison {
    cv::Mat disparity;              // CV_32F, NaN where ours gives no answer
    double ms = 0;                  // our run time
    int sgbmRadius = 1;             // SGBM window radius
    bool occlusionRow = false;      // the method models occlusions: compare how many it finds
    bool answeredErrorRow = false;  // the method leaves holes on purpose: also compare errors where it answers
};
void compareWithSemiGlobal(const Working& w, const OursForComparison& ours, Frame& f, Output& out);

// Click on a left-coordinate view: a box around the pixel on every left view, and on the right view
// the search row and the matched window (or a note when the pixel has no match).
std::vector<Module::ClickShapes> matchClick(const cv::Mat& disparity, int rightView, int viewCount, QPointF px,
                                            const QString& noMatchText);

}  // namespace vl::app::stereo_views
