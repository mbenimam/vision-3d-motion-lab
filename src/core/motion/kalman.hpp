#pragma once
// Constant-velocity Kalman filter for a point in the image: state s = (x, y, vx, vy), one step per frame.
//
//   predict:  s⁻ = F s,               P⁻ = F P Fᵀ + Q          F = [I  I; 0  I] (dt = 1 frame)
//   update:   K = P⁻Hᵀ (H P⁻ Hᵀ + R)⁻¹,  s = s⁻ + K (z − H s⁻),  P = (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ
//
// H reads the position; Q is the discrete white-acceleration model (the object may change its
// velocity by about `processNoise` px/frame each frame); R = measurementNoise² I. Between measurements
// (an occlusion) only predict() runs, so the position follows the last velocity and P grows: the
// covariance ellipse shows how far the object may have gone. The update uses the Joseph form, which
// keeps P symmetric positive definite in floating point. Every step is a few 4×4 products: O(1).

#include <opencv2/core.hpp>

namespace vl::motion {

struct KalmanParams {
    double processNoise = 0.1;      // σ of the acceleration, px / frame²
    double measurementNoise = 2.0;  // σ of a position measurement, px
    double initialSpeedStd = 5.0;   // σ of the unknown initial velocity, px / frame
};

class ConstantVelocityKalman {
public:
    explicit ConstantVelocityKalman(const KalmanParams& p = {});

    void init(cv::Point2d position);
    void predict();
    // z: measured position; noiseScale multiplies R (a less reliable measurement counts less).
    void update(cv::Point2d z, double noiseScale = 1.0);

    cv::Point2d position() const { return {s_[0], s_[1]}; }
    cv::Point2d velocity() const { return {s_[2], s_[3]}; }
    cv::Matx22d positionCovariance() const;
    const cv::Matx44d& covariance() const { return P_; }
    const cv::Vec4d& state() const { return s_; }

private:
    KalmanParams p_;
    cv::Vec4d s_;
    cv::Matx44d P_, F_, Q_;
};

// The nSigma level set of a 2-D Gaussian: semi-axes along the eigenvectors of cov, angle of the
// major axis in degrees (image axes, y down).
struct CovEllipse {
    cv::Point2d centre;
    double major = 0, minor = 0, angleDeg = 0;
};
CovEllipse covarianceEllipse(const cv::Matx22d& cov, cv::Point2d centre, double nSigma = 2.0);

}  // namespace vl::motion
