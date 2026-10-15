#pragma once
// Inputs of the geometry topics. The first sample of each list is synthetic, with exact truth.
//
// Panorama: Inputs::images = 2 or 3 overlapping views; Inputs::truth = one 3x3 CV_64F per image, the
// true map image i -> reference image (index images.size() / 2), or empty when unknown.
//
// Epipolar: Inputs::images = {left, right}; Inputs::truth = {F 3x3 CV_64F with x2ᵀ F x1 = 0,
// N x 4 CV_64F true correspondences (x1, y1, x2, y2)}, or empty when unknown.

#include <vector>

#include "app/module.hpp"

namespace vl::app::geometry_samples {

std::vector<Sample> panorama();
std::vector<Sample> epipolar();

}  // namespace vl::app::geometry_samples
