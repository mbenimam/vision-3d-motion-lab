#include "core/motion/kalman.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace vl::motion {

ConstantVelocityKalman::ConstantVelocityKalman(const KalmanParams& p) : p_(p) {
    F_ = cv::Matx44d::eye();
    F_(0, 2) = F_(1, 3) = 1.0;
    // White acceleration a ~ N(0, q²) over one frame: position moves by a/2, velocity by a.
    const double q2 = p_.processNoise * p_.processNoise;
    Q_ = cv::Matx44d::zeros();
    Q_(0, 0) = Q_(1, 1) = 0.25 * q2;
    Q_(0, 2) = Q_(2, 0) = Q_(1, 3) = Q_(3, 1) = 0.5 * q2;
    Q_(2, 2) = Q_(3, 3) = q2;
    init({0, 0});
}

void ConstantVelocityKalman::init(cv::Point2d position) {
    s_ = {position.x, position.y, 0.0, 0.0};
    const double r2 = p_.measurementNoise * p_.measurementNoise, v2 = p_.initialSpeedStd * p_.initialSpeedStd;
    P_ = cv::Matx44d::diag({r2, r2, v2, v2});
}

void ConstantVelocityKalman::predict() {
    s_ = F_ * s_;
    P_ = F_ * P_ * F_.t() + Q_;
}

void ConstantVelocityKalman::update(cv::Point2d z, double noiseScale) {
    const double r2 = p_.measurementNoise * p_.measurementNoise * std::max(noiseScale, 1e-6);
    // H = [I 0], so H P Hᵀ is the top-left block and P Hᵀ the first two columns of P.
    const cv::Matx22d S(P_(0, 0) + r2, P_(0, 1), P_(1, 0), P_(1, 1) + r2);
    const cv::Matx22d Si = S.inv();
    cv::Matx<double, 4, 2> PHt;
    for (int i = 0; i < 4; ++i) {
        PHt(i, 0) = P_(i, 0);
        PHt(i, 1) = P_(i, 1);
    }
    const cv::Matx<double, 4, 2> K = PHt * Si;
    const cv::Vec2d innovation(z.x - s_[0], z.y - s_[1]);
    s_ += K * innovation;
    cv::Matx44d IKH = cv::Matx44d::eye();
    for (int i = 0; i < 4; ++i) {
        IKH(i, 0) -= K(i, 0);
        IKH(i, 1) -= K(i, 1);
    }
    P_ = IKH * P_ * IKH.t() + r2 * (K * K.t());
}

cv::Matx22d ConstantVelocityKalman::positionCovariance() const {
    return {P_(0, 0), P_(0, 1), P_(1, 0), P_(1, 1)};
}

CovEllipse covarianceEllipse(const cv::Matx22d& c, cv::Point2d centre, double nSigma) {
    // Eigenvalues of a symmetric 2×2 matrix in closed form.
    const double a = c(0, 0), b = 0.5 * (c(0, 1) + c(1, 0)), d = c(1, 1);
    const double mean = 0.5 * (a + d), rad = std::sqrt(0.25 * (a - d) * (a - d) + b * b);
    const double l1 = mean + rad, l2 = std::max(0.0, mean - rad);
    CovEllipse e;
    e.centre = centre;
    e.major = nSigma * std::sqrt(l1);
    e.minor = nSigma * std::sqrt(l2);
    e.angleDeg = 0.5 * std::atan2(2 * b, a - d) * 180.0 / std::numbers::pi;
    return e;
}

}  // namespace vl::motion
