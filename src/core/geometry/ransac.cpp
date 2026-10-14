// RANSAC helpers (the loop itself is the template in ransac.hpp). Course "3D Vision and Motion", part 4.

#include "core/geometry/ransac.hpp"

#include <cmath>

namespace vl::geometry {

int adaptiveTrials(double inlierRatio, int sampleSize, double confidence, int cap) {
    const double w = std::clamp(inlierRatio, 0.0, 1.0);
    const double clean = std::pow(w, sampleSize);  // probability that a sample has no outlier
    if (clean <= 1e-12) return cap;
    if (clean >= 1.0 - 1e-12) return 1;
    // log1p(−x) = log(1 − x) without losing precision when w^s is tiny.
    const double N = std::log(1.0 - std::clamp(confidence, 0.0, 1.0 - 1e-12)) / std::log1p(-clean);
    return int(std::clamp(std::ceil(N), 1.0, double(cap)));
}

std::vector<int> drawSample(std::mt19937& rng, int n, int s) {
    std::uniform_int_distribution<int> pick(0, n - 1);
    std::vector<int> sample;
    while (int(sample.size()) < s) {
        const int i = pick(rng);
        if (std::find(sample.begin(), sample.end(), i) == sample.end()) sample.push_back(i);
    }
    return sample;
}

int countInliers(const std::vector<double>& residuals, double threshold, std::vector<std::uint8_t>& inlier) {
    inlier.resize(residuals.size());
    int count = 0;
    for (size_t i = 0; i < residuals.size(); ++i) {
        inlier[i] = residuals[i] < threshold;  // false for NaN
        count += inlier[i];
    }
    return count;
}

std::vector<int> indicesOf(const std::vector<std::uint8_t>& mask) {
    std::vector<int> idx;
    for (int i = 0; i < int(mask.size()); ++i)
        if (mask[size_t(i)]) idx.push_back(i);
    return idx;
}

}  // namespace vl::geometry
