#pragma once
// Epipolar geometry of two uncalibrated views.
//
// A 3D point X seen at x1 in image 1 and x2 in image 2: the two rays and the baseline are coplanar,
// which in pixel coordinates reads  x2ᵀ F x1 = 0  (convention used throughout this file). F is 3x3,
// rank 2, defined up to scale (7 degrees of freedom). F x1 is the epipolar line of x1 in image 2;
// Fᵀ x2 the line of x2 in image 1. All epipolar lines of image 1 meet at the epipole e1 (F e1 = 0,
// the image of camera 2's centre), those of image 2 at e2 (Fᵀ e2 = 0).
//
// Normalised 8-point algorithm (Hartley 1997): each correspondence gives one linear equation in the
// 9 entries of F, A f = 0; f is the singular vector of A for the smallest singular value. Then
//  - rank 2 is enforced by zeroing the smallest singular value of F (the closest rank-2 matrix in the
//    Frobenius norm). Without it, epipolar lines do not meet in one point.
//  - Coordinates are normalised (centroid 0, mean distance √2) before, and F de-normalised after:
//    F = T2ᵀ F̃ T1. Without it, the monomials x·x' (~10⁶) and 1 differ by six orders of magnitude.
//
// RANSAC scores each correspondence by its Sampson distance: the first-order approximation of the
// geometric (reprojection) error, in pixels,
//   d = |x2ᵀ F x1| / sqrt((F x1)₁² + (F x1)₂² + (Fᵀ x2)₁² + (Fᵀ x2)₂²).

#include <optional>
#include <stop_token>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "core/geometry/homography.hpp"
#include "core/geometry/ransac.hpp"

namespace vl::geometry {

struct EightPointOptions {
    bool normalise = true;
    bool enforceRank2 = true;
};

// n ≥ 8 correspondences (x2ᵀ F x1 = 0), F scaled to unit Frobenius norm. nullopt when degenerate.
std::optional<cv::Matx33d> eightPoint(const Points& x1, const Points& x2, EightPointOptions opt = {});

// Sampson distance (pixels) of one correspondence.
double sampsonDistance(const cv::Matx33d& F, const cv::Point2d& x1, const cv::Point2d& x2);

// Distance (pixels) from a point to the line a x + b y + c = 0.
double pointLineDistance(const cv::Vec3d& line, const cv::Point2d& p);

// Mean over pairs of the distances of x2 to F x1 and of x1 to Fᵀ x2 (symmetric epipolar distance).
double meanSymmetricEpipolarDistance(const cv::Matx33d& F, const Points& x1, const Points& x2);

inline cv::Vec3d lineInImage2(const cv::Matx33d& F, const cv::Point2d& x1) { return F * cv::Vec3d(x1.x, x1.y, 1); }
inline cv::Vec3d lineInImage1(const cv::Matx33d& F, const cv::Point2d& x2) {
    return F.t() * cv::Vec3d(x2.x, x2.y, 1);
}

struct Epipoles {
    cv::Vec3d e1, e2;  // homogeneous: F e1 ≈ 0, Fᵀ e2 ≈ 0 (null vectors, i.e. smallest singular vectors)
};
Epipoles epipoles(const cv::Matx33d& F);

// Finite point of a homogeneous vector, nullopt when it is (numerically) at infinity.
std::optional<cv::Point2d> finitePoint(const cv::Vec3d& h);

// The segment of the line a x + b y + c = 0 inside [0, w] x [0, h], if any.
std::optional<std::pair<cv::Point2d, cv::Point2d>> clipLine(const cv::Vec3d& line, cv::Size size);

// Camera 1 = K1 [I | 0], camera 2 = K2 [R | t]:  F = K2⁻ᵀ [t]× R K1⁻¹.
cv::Matx33d fundamentalFromCameras(const cv::Matx33d& K1, const cv::Matx33d& K2, const cv::Matx33d& R,
                                   const cv::Vec3d& t);

struct FundamentalEstimate {
    cv::Matx33d F = cv::Matx33d::zeros();
    bool ok = false;
    std::vector<std::uint8_t> inliers;
    int nInliers = 0;
    double meanSampson = 0;     // over inliers (pixels)
    cv::Vec3d singularValues;   // of F (unit norm): σ3 = 0 for a true fundamental matrix
    RansacReport report;
};

// RANSAC over 8-point samples scored with the Sampson distance, then an 8-point refit on all inliers.
FundamentalEstimate estimateFundamental(const Points& x1, const Points& x2, const RansacParams& p,
                                        EightPointOptions opt = {}, std::stop_token stop = {});

}  // namespace vl::geometry
