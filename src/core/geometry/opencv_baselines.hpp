#pragma once
// OpenCV's own feature and geometry methods, used as the baselines our implementations are compared
// to. Nothing here is part of the method being taught: thin wrappers that run OpenCV and return its
// answer in our conventions (Points, x2ᵀ F x1 = 0, 0/1 inlier masks) plus its run time.
//
//   siftMatches          the full OpenCV front end: cv::SIFT, cv::BFMatcher::knnMatch (k = 2), ratio test
//   findHomography       cv::findHomography(RANSAC) on given correspondences
//   findFundamental      cv::findFundamentalMat(FM_RANSAC) on given correspondences
//
// Feeding the last two with *our* matches compares the estimators on identical data.

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "core/geometry/homography.hpp"

namespace vl::geometry::opencv {

struct Correspondences {
    Points a, b;                        // a[i] <-> b[i]
    int keypointsA = 0, keypointsB = 0;
    double ms = 0;                      // detection + description + matching
};

// cv::SIFT (at most maxFeatures per image) on both images, cv::BFMatcher (L2) 2-nearest neighbours,
// kept when nearest / second nearest < ratio.
Correspondences siftMatches(const cv::Mat& imageA, const cv::Mat& imageB, float ratio, int maxFeatures);

// Keypoint count of cv::SIFT and where they are (for repeatability tests).
std::vector<cv::Point2d> siftKeypoints(const cv::Mat& image, int maxFeatures);

struct Homography {
    cv::Matx33d H = cv::Matx33d::eye();
    bool ok = false;
    std::vector<std::uint8_t> inliers;
    int nInliers = 0;
    double ms = 0;
};
Homography findHomography(const Points& x, const Points& xp, double threshold, double confidence);

struct Fundamental {
    cv::Matx33d F = cv::Matx33d::zeros();  // x2ᵀ F x1 = 0, unit Frobenius norm
    bool ok = false;
    std::vector<std::uint8_t> inliers;
    int nInliers = 0;
    double ms = 0;
};
Fundamental findFundamental(const Points& x1, const Points& x2, double threshold, double confidence);

}  // namespace vl::geometry::opencv
