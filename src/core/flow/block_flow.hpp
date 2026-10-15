#pragma once
// Block-matching optical flow, the simplest dense estimator. Course: "3D Vision and Motion",
// part 11 (and part 6 for the same idea in stereo, where the search is along a row).
//
// For each pixel x of frame 0, compare its (2r+1)² block with the block of frame 1 displaced by
// d = (dx, dy), for every d of a (2S+1)² search window, and keep the d of least cost:
//     C(x, d) = mean over the block of |I0(x') − I1(x' + d)|   (SAD)   or   (I0(x') − I1(x' + d))²   (SSD)
//     w(x) = argmin_d C(x, d)
// No linearisation, so no limit on the size of the motion other than S; but flat areas and straight
// edges give ambiguous minima (the aperture problem, with no smoothness to fill in), and the answer
// is a whole number of pixels unless refined. Cost O(W·H·(2S+1)²) whatever the block size (the block
// sums come from an integral image).

#include <stop_token>

#include <opencv2/core.hpp>

#include "core/common/task.hpp"

namespace vl::flow {

enum class BlockCost { SAD, SSD };

struct BlockFlowParams {
    int radius = 4;         // block is (2r+1)²
    int search = 8;         // displacements in [-S, S]²
    BlockCost cost = BlockCost::SAD;
    bool subpixel = true;
};

struct BlockFlowResult {
    cv::Mat flow;   // CV_32FC2
    cv::Mat cost;   // CV_32F, best cost per pixel, mean per block pixel (grey levels, or grey² for SSD)
    long evaluations = 0;  // number of (pixel, displacement) costs computed
};

// frame0, frame1: 8-bit grey or BGR images of the same size.
BlockFlowResult blockMatchFlow(const cv::Mat& frame0, const cv::Mat& frame1, const BlockFlowParams& p,
                               std::stop_token stop = {}, const ProgressFn& progress = {});

// The cost of every displacement of the search window for one pixel: a (2S+1)² CV_32F image whose
// centre is zero motion (row = dy + S, column = dx + S); infinite where x + d leaves frame 1.
// I0, I1: grey CV_32F frames. O((2S+1)²·(2r+1)²); meant for one clicked pixel.
cv::Mat blockCostSurface(const cv::Mat& I0, const cv::Mat& I1, cv::Point p, const BlockFlowParams& params);

}  // namespace vl::flow
