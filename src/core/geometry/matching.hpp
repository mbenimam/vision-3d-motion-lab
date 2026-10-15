#pragma once
// Putative correspondences between two sets of descriptors. Course "3D Vision and Motion", part 3.
//
// Steps (matching.cpp):
//   1. distanceMatrix     d²(i, j) = ‖a_i − b_j‖² for every pair (brute force)
//   2. twoNearest         per row i: the nearest b_j and the second nearest
//   3. ratio test         keep i ↔ j when d₁ / d₂ < ratio (Lowe): with repetitive texture two
//                         candidates look alike and the match is ambiguous, so it is dropped
//   4. mutual check       (optional) a_i must also be the nearest of b_j among all a
//
// Cost O(n_a · n_b · 128) operations and n_a · n_b floats of memory (36 MB for 3000 × 3000).

#include <stop_token>
#include <vector>

#include <opencv2/core.hpp>

namespace vl::geometry {

struct Match {
    int a = -1, b = -1;   // indices in the first / second descriptor set
    float distance = 0;   // ‖a − b‖ to the nearest neighbour
    float ratio = 0;      // nearest / second nearest
};

struct MatchParams {
    float ratio = 0.75f;  // Lowe's threshold; 1 disables the test
    bool mutual = true;
};

// da, db: CV_32F, one descriptor per row, same width.
std::vector<Match> matchDescriptors(const cv::Mat& da, const cv::Mat& db, const MatchParams& p,
                                    std::stop_token stop = {});

}  // namespace vl::geometry
