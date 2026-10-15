#pragma once
// Kalman filter for a point moving in the image at constant velocity. Course: "3D Vision and
// Motion", part 10 (predictive filtering).
//
// The state Θ = (x, y, vx, vy) evolves linearly with Gaussian noise, and we measure its position:
//
//     Θ_t = A Θ_{t−1} + W,   W ~ N(0, Q)          A = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1]  (δt = 1 frame)
//     X_t = H Θ_t + V,       V ~ N(0, R)          H = [1 0 0 0; 0 1 0 0],  R = σ_m² I
//
// One frame is two steps, one function each:
//   1. predict   Θ⁻ = A Θ,                     P⁻ = A P Aᵀ + Q
//   2. update    K = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹,    Θ = Θ⁻ + K (X − H Θ⁻),    P = (I − K H) P⁻
//
// Q is the discrete white-acceleration model: the object may change its velocity by about
// `processNoise` px/frame each frame. During an occlusion only predict() runs: the position follows
// the last velocity and P grows, and its 2×2 position block, drawn as an ellipse, shows how far the
// object may have gone. Every step is a few 4×4 products: O(1).

#include <opencv2/core.hpp>

namespace vl::motion {

struct KalmanParams {
    double processNoise = 0.1;      // σ_a of the acceleration, px / frame²
    double measurementNoise = 2.0;  // σ_m of a position measurement, px
    double initialSpeedStd = 5.0;   // σ of the unknown initial velocity, px / frame
};

class ConstantVelocityKalman {
public:
    explicit ConstantVelocityKalman(const KalmanParams& p = {});

    // Θ_0 = (position, 0, 0); P_0 = diag(σ_m², σ_m², σ_v², σ_v²).
    void init(cv::Point2d position);
    void predict();
    // X: measured position; noiseScale multiplies R (a less reliable measurement counts less).
    void update(cv::Point2d X, double noiseScale = 1.0);

    cv::Point2d position() const { return {theta_[0], theta_[1]}; }
    cv::Point2d velocity() const { return {theta_[2], theta_[3]}; }
    cv::Matx22d positionCovariance() const;  // H P Hᵀ
    const cv::Vec4d& state() const { return theta_; }
    const cv::Matx44d& covariance() const { return P_; }

    // The model's matrices, e.g. to set up another filter identically.
    const cv::Matx44d& A() const { return A_; }
    const cv::Matx<double, 2, 4>& H() const { return H_; }
    const cv::Matx44d& Q() const { return Q_; }
    cv::Matx22d R() const;

private:
    KalmanParams p_;
    cv::Vec4d theta_;  // Θ
    cv::Matx44d P_;
    cv::Matx44d A_, Q_;
    cv::Matx<double, 2, 4> H_;
};

// The nSigma level set of a 2-D Gaussian: semi-axes along the eigenvectors of cov, angle of the
// major axis in degrees (image axes, y down).
struct CovEllipse {
    cv::Point2d centre;
    double major = 0, minor = 0, angleDeg = 0;
};
CovEllipse covarianceEllipse(const cv::Matx22d& cov, cv::Point2d centre, double nSigma = 2.0);

}  // namespace vl::motion
