#pragma once
// Pieces shared by the geometry topics: matching two images through the feature cache, laying
// images side by side, and turning matches and epipolar lines into overlays.

#include <stop_token>
#include <vector>

#include <opencv2/core.hpp>

#include "app/module.hpp"
#include "core/geometry/features.hpp"
#include "core/geometry/homography.hpp"

namespace vl::app::geometry_common {

inline QPointF qp(const cv::Point2d& p) { return {p.x, p.y}; }

// Putative correspondences between two images: matched keypoint positions, a[i] <-> b[i].
struct Correspondences {
    vl::geometry::Points a, b;
    int featuresA = 0, featuresB = 0;
};
Correspondences correspond(vl::geometry::FeatureCache& cache, const cv::Mat& imgA, const cv::Mat& imgB,
                           const vl::geometry::MatchParams& mp, int maxFeatures, std::stop_token stop);

// Images next to each other on the app background, top-aligned, with a gap; offsets[i] is where
// image i's origin lands.
struct Strip {
    cv::Mat image;
    std::vector<cv::Point2d> offsets;
};
Strip sideBySide(const std::vector<cv::Mat>& images, int gap = 24);

// Lines between matched points of a strip: outliers first (thin, translucent bad), inliers on top
// (good). At most maxOutliers / maxInliers lines of each kind are drawn (evenly subsampled) so the
// picture stays legible; the counts in the stats are always the full ones.
void addMatchLines(std::vector<Shape>& out, const vl::geometry::Points& a, const vl::geometry::Points& b,
                   cv::Point2d offA, cv::Point2d offB, const std::vector<std::uint8_t>& inliers, bool showOutliers,
                   int maxOutliers = 150, int maxInliers = 250);

// The visible part of a line a x + b y + c = 0 inside an image of the given size, shifted by offset.
std::optional<LineShape> lineShape(const cv::Vec3d& line, cv::Size size, cv::Point2d offset, Pen pen);

// n well-spread indices among those with mask != 0 (greedy farthest-point sampling on pts).
std::vector<int> spreadSubset(const vl::geometry::Points& pts, const std::vector<std::uint8_t>& mask, int n);

// A distinct, bright colour for item i (for pairing a point with its line across views).
QColor pairColor(int i);

inline const cv::Scalar kBackground(23, 17, 14);  // #0e1117, the canvas background

}  // namespace vl::app::geometry_common
