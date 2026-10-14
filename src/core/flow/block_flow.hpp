#pragma once
// Block-matching flow: the simplest dense estimator. For each pixel of frame 0, compare its
// (2r+1)² block with every block of frame 1 displaced by (dx, dy) in a (2S+1)² search window and
// keep the displacement with the lowest cost (SAD or SSD of grey levels).
//
// Like the stereo version, the cost of one displacement is computed for the whole image and summed
// over the block with a box filter, so the price is O(W·H·(2S+1)²) whatever the block size. Integer
// by nature; the optional sub-pixel step fits a parabola through the costs of the 4 neighbouring
// displacements, separately in x and y.
//
// Its failure modes are the ones the variational methods address: flat areas and edges give
// ambiguous minima (the aperture problem, no smoothness to fill in), and motion larger than S is
// simply not searched.

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
// centre is zero motion (row = dy + S, column = dx + S). Blocks are clipped at the image border.
// I0, I1: grey CV_32F frames. O((2S+1)²·(2r+1)²); meant for one clicked pixel.
cv::Mat blockCostSurface(const cv::Mat& I0, const cv::Mat& I1, cv::Point p, const BlockFlowParams& params);

}  // namespace vl::flow
