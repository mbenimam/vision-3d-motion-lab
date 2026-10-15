#pragma once
// Showing a flow field. Two standard pictures:
//  - the Middlebury colour code (Baker et al. 2011): the direction of a vector is a hue on a colour
//    wheel, its length the saturation (white = no motion, full colour = maxMagnitude);
//  - arrows subsampled on a grid, scaled up so that the longest one fills about one grid step.
//
// Axis directions on the wheel (image coordinates, y down): right = red, down = yellow-orange,
// left = cyan, up = blue-violet.

#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace vl::flow {

// Colour of one vector (u, v) for a field normalised by maxMagnitude. Vectors longer than
// maxMagnitude keep their hue but are darkened, as in the Middlebury code. BGR, 8-bit.
cv::Vec3b flowColor(float u, float v, float maxMagnitude);

// The whole field as an 8-bit BGR image. NaN vectors are black.
cv::Mat flowToColor(const cv::Mat& flow, float maxMagnitude);

// Reference wheel of the code: a size.width x size.height dark image with a disc of diameter
// min(size) - 2·margin centred in it; the disc edge is maxMagnitude. Use wheelPosition() to place a
// vector on it.
cv::Mat colorWheel(cv::Size size, float maxMagnitude, int margin = 24);
cv::Point2f wheelPosition(cv::Size size, float maxMagnitude, cv::Point2f vec, int margin = 24);

// A robust maximum for the colour scale: the 99th percentile of |w| over finite vectors, at least 1 px.
float robustMaxMagnitude(const cv::Mat& flow);

struct Arrow {
    cv::Point2f from, to;
};

// One arrow every `step` pixels (centred in each cell), drawn `scale` times longer than the vector.
// NaN vectors and vectors shorter than minLength (before scaling) are skipped.
std::vector<Arrow> flowArrows(const cv::Mat& flow, int step, float scale, float minLength = 0.05f);

// Scale that makes the 95th-percentile vector about 0.9 grid steps long (at least 1).
float arrowScale(const cv::Mat& flow, int step);

}  // namespace vl::flow
