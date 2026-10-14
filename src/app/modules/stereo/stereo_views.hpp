#pragma once
// Pictures and numbers shared by the seed-propagation and graph-cut topics: the evaluation mask,
// the disparity and error views, and the click that shows a pixel's match in the right image.

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

// Bad-pixel share (|d - truth| > 1 px, NaN counts as wrong) on the visible pixels with known truth.
double badRate(const cv::Mat& disparity, const Working& w);

// The left image greyed out; on visible pixels with known truth: wrong disparities in pink, NaN in blue.
cv::Mat errorImage(const cv::Mat& disparity, const Working& w);

// Left image at half brightness, in grey, ready for colour overlays.
cv::Mat dimmed(const cv::Mat& bgr);

// Click on a left-coordinate view: a box around the pixel on every left view, and on the right view
// the search row and the matched window (or a note when the pixel has no match).
std::vector<Module::ClickShapes> matchClick(const cv::Mat& disparity, int rightView, int viewCount, QPointF px,
                                            const QString& noMatchText);

}  // namespace vl::app::stereo_views
