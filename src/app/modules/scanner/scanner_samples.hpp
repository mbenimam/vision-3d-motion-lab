#pragma once
// Inputs of the document scanner: synthetic photos of a page on a desk, with exact ground truth.
//
// Inputs::images = {photo}; Inputs::truth = {corners (4×2 CV_64F, rows TL, TR, BR, BL, photo pixels),
// the flat page (8-bit BGR)}. A photo the user opens has no truth.

#include <optional>
#include <vector>

#include "app/module.hpp"
#include "core/scanner/quad.hpp"

namespace vl::app::scanner_samples {

std::vector<Sample> all();

std::optional<vl::scanner::Quad> trueCorners(const Inputs& in);
cv::Mat truePage(const Inputs& in);  // empty when unknown

}  // namespace vl::app::scanner_samples
