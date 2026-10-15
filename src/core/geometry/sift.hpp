#pragma once
// SIFT keypoints and descriptors (Lowe 2004). Course "3D Vision and Motion", part 3.
//
// A keypoint is a blob or corner found at its own scale, so the same physical point is found again
// when the image is zoomed, turned or seen under another light. The steps are listed in sift.cpp.
//
// Choice made here: like Lowe and OpenCV, the image is first doubled (most keypoints of a detailed
// photo are at the finest scales: ~2-8x more keypoints). It costs ~4x the time; `upsample = false`
// processes the image at its own resolution instead.

#include <stop_token>
#include <vector>

#include <opencv2/core.hpp>

namespace vl::geometry {

struct Keypoint {
    cv::Point2d pt;       // position in the input image (pixels)
    double sigma = 0;     // scale σ of the blob, in input-image pixels
    double angle = 0;     // dominant gradient orientation, radians in [0, 2π), y pointing down
    double contrast = 0;  // |D(x̂)| at the refined extremum (image values in [0, 1])
    int octave = 0;       // where it was found: octave o, layer i + ŝ inside the octave
    double layer = 0;
};

struct Features {
    std::vector<Keypoint> keypoints;
    cv::Mat descriptors;  // CV_32F, one 128-d unit row per keypoint
};

struct SiftParams {
    int intervals = 3;               // s: layers per octave where extrema are searched
    double sigma0 = 1.6;             // σ₀: blur of the first layer of each octave
    double contrastThreshold = 0.04; // reject |D(x̂)| < contrastThreshold / s (DoG values scale with 1/s)
    double edgeRatio = 10;           // r: reject when the principal curvatures differ by more than r
    bool upsample = true;            // double the image first (more small keypoints, ~4x slower)
    int maxFeatures = 3000;          // keep the strongest (by contrast); 0 keeps all
};

// The whole pipeline on an 8-bit grey or BGR image.
Features detectSift(const cv::Mat& image, const SiftParams& p = {}, std::stop_token stop = {});

}  // namespace vl::geometry
