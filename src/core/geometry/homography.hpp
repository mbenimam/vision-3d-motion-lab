#pragma once
// Plane-to-plane projective maps. Two photos of a planar scene, or two photos taken by a camera that
// only rotates about its centre, are related by a 3x3 homography: λ x' = H x (H has 8 degrees of freedom).
//
// Direct Linear Transform: x' × (H x) = 0 gives two independent linear equations in the 9 entries of
// H per correspondence, A h = 0. With n ≥ 4 points, h is the right singular vector of A for the
// smallest singular value (min ‖A h‖ subject to ‖h‖ = 1).
//
// Hartley normalisation: pixel coordinates (~10³) make the entries of A range from 1 to 10⁶, so A is
// badly conditioned and the algebraic error weighs the equations unevenly. Translating each point set
// to its centroid and scaling it to a mean distance of √2 fixes that; the result is de-normalised by
// H = T'⁻¹ H̃ T.

#include <optional>
#include <stop_token>
#include <vector>

#include <opencv2/core.hpp>

#include "core/geometry/ransac.hpp"

namespace vl::geometry {

using Points = std::vector<cv::Point2d>;

// Similarity T (scale + translation) taking the points to centroid 0 and mean distance √2.
cv::Matx33d hartleyNormalisation(const Points& pts);

// Projective map of one point (no check for points at infinity: w = 0 gives inf).
inline cv::Point2d transform(const cv::Matx33d& H, const cv::Point2d& p) {
    const double w = H(2, 0) * p.x + H(2, 1) * p.y + H(2, 2);
    return {(H(0, 0) * p.x + H(0, 1) * p.y + H(0, 2)) / w, (H(1, 0) * p.x + H(1, 1) * p.y + H(1, 2)) / w};
}

// DLT from n ≥ 4 correspondences src[i] -> dst[i], scaled so H(2,2) = 1 when possible.
// Exact for 4 points in general position; least squares (algebraic) for more. nullopt when degenerate.
std::optional<cv::Matx33d> dltHomography(const Points& src, const Points& dst, bool normalise = true);

// |H src - dst| for each pair (one-way transfer error, pixels).
void transferErrors(const cv::Matx33d& H, const Points& src, const Points& dst, std::vector<double>& out);

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

// RANSAC over 4-point DLT samples with the transfer error, then a DLT refit on the inliers.
HomographyEstimate estimateHomography(const Points& src, const Points& dst, const RansacParams& p,
                                      bool normalise = true, std::stop_token stop = {});

}  // namespace vl::geometry
