#pragma once
// Plane-to-plane projective maps. Course "3D Vision and Motion", part 2 (homographies, DLT,
// panoramas), estimated robustly with RANSAC (part 4).
//
// Two photos of a planar scene, or two photos taken by a camera that only turns about its centre, are
// related by a 3×3 homography: λ x' = H x (8 degrees of freedom, H up to scale). Steps (homography.cpp):
//   1. hartleyNormalisation  T: centroid to 0, mean distance to √2, for each point set
//   2. dltHomography         x' × (H x) = 0 gives two linear equations in h per correspondence, A h = 0;
//                            h = right singular vector of A for the smallest singular value (min ‖A h‖,
//                            ‖h‖ = 1); de-normalised by H = T'⁻¹ H̃ T
//   3. transferErrors        residual ‖H x − x'‖ of each correspondence (pixels)
//   4. estimateHomography    RANSAC over 4-point DLT samples, then DLT on all inliers
//
// Why normalise: in pixels (~10³) the entries of A range from 1 to 10⁶, so A is badly conditioned
// and the algebraic error weighs the equations unevenly.

#include <cstdint>
#include <optional>
#include <stop_token>
#include <vector>

#include <opencv2/core.hpp>

#include "core/geometry/ransac.hpp"

namespace vl::geometry {

using Points = std::vector<cv::Point2d>;

// Step 1: similarity T (scale + translation) taking the points to centroid 0 and mean distance √2.
cv::Matx33d hartleyNormalisation(const Points& pts);

// x' = H x for one point (w = 0, a point at infinity, gives inf).
inline cv::Point2d transform(const cv::Matx33d& H, const cv::Point2d& x) {
    const double w = H(2, 0) * x.x + H(2, 1) * x.y + H(2, 2);
    return {(H(0, 0) * x.x + H(0, 1) * x.y + H(0, 2)) / w, (H(1, 0) * x.x + H(1, 1) * x.y + H(1, 2)) / w};
}

// Step 2: DLT from n ≥ 4 correspondences x[i] -> xp[i], scaled so H(2,2) = 1 when possible. Exact for
// 4 points in general position, least squares (algebraic) for more. nullopt when degenerate.
std::optional<cv::Matx33d> dltHomography(const Points& x, const Points& xp, bool normalise = true);

// Step 3: ‖H x − x'‖ for each pair (one-way transfer error, pixels).
void transferErrors(const cv::Matx33d& H, const Points& x, const Points& xp, std::vector<double>& out);

// Mean distance between the image corners mapped by H and by the true map (pixels): a
// parameter-free way to compare two homographies over the image they act on.
double cornerError(const cv::Matx33d& H, const cv::Matx33d& truth, cv::Size size);

// True when three of the four points are (nearly) collinear: such a sample cannot define H.
bool collinearTriple(const cv::Point2d* p);

struct HomographyEstimate {
    cv::Matx33d H = cv::Matx33d::eye();  // maps the first point set onto the second
    bool ok = false;
    std::vector<std::uint8_t> inliers;
    int nInliers = 0;
    double rms = 0;                      // transfer error RMS over inliers (pixels)
    RansacReport report;
};

// Step 4.
HomographyEstimate estimateHomography(const Points& x, const Points& xp, const RansacParams& p,
                                      bool normalise = true, std::stop_token stop = {});

}  // namespace vl::geometry
