#pragma once
// Synthetic scenes with exact ground truth for the geometry topics, and point-only generators for
// the tests.

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "core/geometry/homography.hpp"

namespace vl::geometry::scenes {

// A camera turning about its centre in front of a painted backdrop: view i = K R_i K⁻¹ applied to the
// backdrop, so view i -> view j is exactly H = K R_j R_iᵀ K⁻¹. Each view gets its own exposure (gain)
// so that seams show when blending is off.
struct RotatingViews {
    std::vector<cv::Mat> images;          // 8-bit BGR
    std::vector<cv::Matx33d> toReference; // exact map view i -> view `reference`
    int reference = 0;
};
RotatingViews rotatingCamera(cv::Size viewSize, int nViews, double stepDeg, std::uint32_t seed);

// Two pinhole cameras in a textured room (floor, two walls, a box): camera 1 = K [I | 0],
// camera 2 = K [R | t], rendered by ray casting, with the exact F and correspondences of visible points.
struct TwoViewScene {
    cv::Mat left, right;     // 8-bit BGR
    cv::Matx33d K, R;
    cv::Vec3d t;
    cv::Matx33d F;           // x2ᵀ F x1 = 0, unit norm
    Points x1, x2;           // true correspondences of points visible in both views
};
TwoViewScene room(cv::Size size, std::uint32_t seed);

// Random correspondences for tests: inliers exactly related by a model plus Gaussian noise (pixels),
// outliers uniform over the image. isInlier marks which is which.
struct PointPairs {
    Points a, b;
    std::vector<std::uint8_t> isInlier;
    Points cleanA, cleanB;   // noise-free inliers
};
PointPairs homographyPairs(const cv::Matx33d& H, cv::Size size, int nInliers, int nOutliers, double noise,
                           std::uint32_t seed);

// Random 3D points in front of two cameras (K, [I|0]) and (K, [R|t]).
PointPairs twoViewPairs(const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t, cv::Size size, int nInliers,
                        int nOutliers, double noise, std::uint32_t seed);

// Rotation from yaw (about y), pitch (about x), roll (about z), degrees: R = Rz Rx Ry.
cv::Matx33d rotation(double yawDeg, double pitchDeg, double rollDeg);

}  // namespace vl::geometry::scenes
