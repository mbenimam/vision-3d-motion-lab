#pragma once
// Seed-and-grow stereo: trust only the best matches first, then let them spread to their neighbours,
// best first. See seeds.cpp for the steps.

#include <functional>
#include <stop_token>
#include <vector>

#include <opencv2/core.hpp>

#include "core/common/task.hpp"

namespace vl::stereo {

struct SeedParams {
    int minDisparity = 0;
    int maxDisparity = 48;       // inclusive
    int radius = 3;              // ZNCC window is (2r+1)²
    double seedScore = 0.95;     // ZNCC a match needs to be a seed
    bool leftRightCheck = true;  // seeds must also be the right image's best match back
    int seedSpacing = 1;         // > 1: keep only the best seed of each spacing × spacing cell
    double stopScore = 0.6;      // growth does not accept matches below this ZNCC
};

struct SeedResult {
    cv::Mat disparity;  // CV_32F, left coordinates, NaN where never reached
    cv::Mat score;      // CV_32F, ZNCC of the kept match, NaN where never reached
    cv::Mat order;      // CV_32S, 0 for seeds, k >= 1 for the k-th pixel reached by growth, -1 never
    std::vector<cv::Point> seeds;
    int grown = 0;      // pixels reached by growth (not counting seeds)
};

// Called every `every` grown pixels with the current disparity map and the number of pixels reached.
using SeedObserver = std::function<void(const cv::Mat& disparity, const cv::Mat& order, int reached)>;

// left, right: rectified 8-bit grey or BGR images of the same size; left (x, y) matches right (x − d, y).
SeedResult propagateSeeds(const cv::Mat& left, const cv::Mat& right, const SeedParams& p, std::stop_token stop = {},
                          const ProgressFn& progress = {}, const SeedObserver& observer = {}, int every = 0);

}  // namespace vl::stereo
