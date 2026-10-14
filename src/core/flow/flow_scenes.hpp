#pragma once
// Synthetic flow pairs that isolate one difficulty (vl::synth::rigidFlow covers smooth motion).

#include <cstdint>

#include <opencv2/core.hpp>

#include "core/common/synthetic.hpp"

namespace vl::flow {

// A flat, untextured square sliding by `shift` over a static, low-contrast textured background.
// Inside the square there is no gradient at all, along its sides only the normal motion is visible
// (aperture problem), only its corners pin the full motion down; its outline is a motion boundary. Rendered with exact
// area coverage, so sub-pixel shifts are honest. Truth: `shift` where frame 0 shows the square, 0 elsewhere.
vl::synth::FlowPair slidingSquare(cv::Size size, cv::Rect2d square, cv::Point2d shift, std::uint32_t seed);

}  // namespace vl::flow
