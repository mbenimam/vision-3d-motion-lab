#include "core/geometry/ransac.hpp"

#include <algorithm>
#include <cmath>

namespace vl::geometry {

int adaptiveTrials(double inlierRatio, int sampleSize, double confidence, int cap) {
    const double w = std::clamp(inlierRatio, 0.0, 1.0);
    const double clean = std::pow(w, sampleSize);  // probability that a sample has no outlier
    if (clean <= 1e-12) return cap;
    if (clean >= 1.0 - 1e-12) return 1;
    // log1p keeps precision when w^s is tiny (the case the course warns about).
    const double n = std::log(1.0 - std::clamp(confidence, 0.0, 1.0 - 1e-12)) / std::log1p(-clean);
    return static_cast<int>(std::clamp(std::ceil(n), 1.0, double(cap)));
}

}  // namespace vl::geometry
