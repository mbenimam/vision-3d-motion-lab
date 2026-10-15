#pragma once
// Inputs shared by the stereo topics: a synthetic scene with exact ground truth (always there)
// and the Middlebury pairs when scripts/fetch_data.sh has been run.
//
// Inputs::images = {left, right}; Inputs::truth = {disparity CV_32F (0 = unknown),
// visible mask CV_8U (255 where the left pixel is seen in the right view; derived from the truth for Middlebury)}.
// Middlebury disparities are positive with left (x) matching right (x - d), like ours.

#include <vector>

#include "app/module.hpp"

namespace vl::app::stereo_samples {

std::vector<Sample> all();
int maxDisparityHint(const Inputs& in);  // a sensible search range for this pair

}  // namespace vl::app::stereo_samples
