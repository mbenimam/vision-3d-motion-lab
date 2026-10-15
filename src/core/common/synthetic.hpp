#pragma once
// Synthetic inputs with exact ground truth. They make the app useful without any download,
// and they are what the unit tests measure against.

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

namespace vl::synth {

// Grey 8-bit texture rich enough to match on: multi-octave value noise plus a few strokes.
cv::Mat texture(cv::Size size, std::uint32_t seed);

// Same, but in colour (three independent noise fields blended), 8-bit BGR.
cv::Mat colorTexture(cv::Size size, std::uint32_t seed);

struct StereoScene {
    cv::Mat left, right;   // 8-bit BGR, rectified: left (x, y) matches right (x - d, y)
    cv::Mat disparity;     // CV_32F, ground truth in left coordinates
    cv::Mat occluded;      // CV_8U, 255 where the left pixel is hidden in the right view
    int maxDisparity = 0;
};

// Layered scene: a far slanted wall, a box and a disc in front. Disparities in [minD, maxD].
StereoScene layeredStereo(cv::Size size, int maxDisparity, std::uint32_t seed);

struct FlowPair {
    cv::Mat frame0, frame1;  // 8-bit BGR
    cv::Mat flow;            // CV_32FC2, frame0 (x, y) moves to frame1 (x + u, y + v)
};

// A smooth flow field: rotation by angleDeg and zoom about the centre, plus a translation.
FlowPair rigidFlow(cv::Size size, double angleDeg, double zoom, cv::Point2d shift, std::uint32_t seed);

struct MovingScene {
    std::vector<cv::Mat> frames;      // 8-bit BGR
    std::vector<cv::Mat> masks;       // CV_8U, 255 on the moving object
    std::vector<cv::Rect> boxes;      // object bounding box per frame
};

// Static textured background with sensor noise and slow lighting drift, and one textured
// object moving along a Lissajous path. Good for background subtraction and tracking.
MovingScene movingObject(cv::Size size, int nFrames, std::uint32_t seed);

// Add Gaussian noise of standard deviation sigma (grey levels) in place, deterministically from rng.
void addNoise(cv::Mat& img8u, double sigma, cv::RNG& rng);

// Backward warp: out(x) = img(x + flow(x)), bilinear, border replicated.
cv::Mat warpBackward(const cv::Mat& img, const cv::Mat& flow);

}  // namespace vl::synth
