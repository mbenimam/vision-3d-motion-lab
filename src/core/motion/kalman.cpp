// Constant-velocity Kalman filter (see kalman.hpp for the model). Course: "3D Vision and Motion",
// part 10.

#include "core/motion/kalman.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace vl::motion {

ConstantVelocityKalman::ConstantVelocityKalman(const KalmanParams& p) : p_(p) {
    // A: position += velocity · δt, velocity unchanged.
    A_ = cv::Matx44d(1, 0, 1, 0,
                     0, 1, 0, 1,
                     0, 0, 1, 0,
                     0, 0, 0, 1);
    // H: we observe the position only.
    H_ = cv::Matx<double, 2, 4>(1, 0, 0, 0,
                                0, 1, 0, 0);
    // Q = G G ᵀ σ_a² with G = (½, ½, 1, 1)ᵀ per axis: an acceleration a during one frame moves the
    // position by a/2 and the velocity by a, so their noises are correlated.
    const double q = p_.processNoise * p_.processNoise;
    Q_ = cv::Matx44d(q / 4, 0,     q / 2, 0,
                     0,     q / 4, 0,     q / 2,
                     q / 2, 0,     q,     0,
                     0,     q / 2, 0,     q);
    init({0, 0});
}

cv::Matx22d ConstantVelocityKalman::R() const {
    const double r = p_.measurementNoise * p_.measurementNoise;
    return {r, 0, 0, r};
}

void ConstantVelocityKalman::init(cv::Point2d position) {
    theta_ = {position.x, position.y, 0.0, 0.0};
    const double r = p_.measurementNoise * p_.measurementNoise, v = p_.initialSpeedStd * p_.initialSpeedStd;
    P_ = cv::Matx44d::diag({r, r, v, v});
}

// Step 1:  Θ⁻ = A Θ,   P⁻ = A P Aᵀ + Q
void ConstantVelocityKalman::predict() {
    theta_ = A_ * theta_;
    P_ = A_ * P_ * A_.t() + Q_;
}

// Step 2:  K = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹,   Θ = Θ⁻ + K (X − H Θ⁻),   P = (I − K H) P⁻
void ConstantVelocityKalman::update(cv::Point2d X, double noiseScale) {
    const cv::Matx22d R = this->R() * std::max(noiseScale, 1e-6);
    const cv::Matx22d S = H_ * P_ * H_.t() + R;  // covariance of the innovation
    const cv::Matx<double, 4, 2> K = P_ * H_.t() * S.inv();
    const cv::Vec2d innovation = cv::Vec2d(X.x, X.y) - H_ * theta_;
    theta_ = theta_ + K * innovation;
    P_ = (cv::Matx44d::eye() - K * H_) * P_;
}

cv::Matx22d ConstantVelocityKalman::positionCovariance() const { return H_ * P_ * H_.t(); }

CovEllipse covarianceEllipse(const cv::Matx22d& c, cv::Point2d centre, double nSigma) {
    // Eigenvalues of the symmetric 2×2 matrix [a b; b d] in closed form: mean ± √(((a − d)/2)² + b²).
    const double a = c(0, 0), b = 0.5 * (c(0, 1) + c(1, 0)), d = c(1, 1);
    const double mean = 0.5 * (a + d), radius = std::sqrt(0.25 * (a - d) * (a - d) + b * b);
    const double l1 = mean + radius, l2 = std::max(0.0, mean - radius);
    CovEllipse e;
    e.centre = centre;
    e.major = nSigma * std::sqrt(l1);
    e.minor = nSigma * std::sqrt(l2);
    // Direction of the major eigenvector: tan 2φ = 2b / (a − d).
    e.angleDeg = 0.5 * std::atan2(2 * b, a - d) * 180.0 / std::numbers::pi;
    return e;
}

}  // namespace vl::motion
