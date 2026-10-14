#pragma once
// Synthetic sequences for the motion topics, beyond vl::synth::movingObject.

#include <cstdint>
#include <vector>

#include "core/common/synthetic.hpp"

namespace vl::motion {

// A warm textured disc crosses the frame at constant velocity and passes behind a static grey pillar
// in the middle, where it is fully hidden for a few frames. masks[t] is the *visible* part (what
// background subtraction should find); boxes[t] is the whole object's box, hidden or not (where a
// tracker should be). Made to show what a constant-velocity prediction buys during an occlusion.
synth::MovingScene occludedCrossing(cv::Size size, int nFrames, std::uint32_t seed);

// Add white Gaussian sensor noise (σ grey levels per channel, deterministic for a seed) in place.
void addSensorNoise(std::vector<cv::Mat>& frames, double sigma, std::uint32_t seed);

}  // namespace vl::motion
