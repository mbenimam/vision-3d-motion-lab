#pragma once
// Test helper: the synthetic sequence with sensor noise added on top, so the measured numbers do
// not depend on how noisy the generator itself is.

#include "core/common/synthetic.hpp"
#include "core/motion/sequences.hpp"

namespace vl::motion::test {

inline vl::synth::MovingScene noisyMovingObject(cv::Size size, int frames, std::uint32_t seed, double sigma = 3.0) {
    auto s = vl::synth::movingObject(size, frames, seed);
    addSensorNoise(s.frames, sigma, seed);
    return s;
}

}  // namespace vl::motion::test
