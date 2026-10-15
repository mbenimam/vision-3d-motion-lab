#pragma once
// Pieces shared by the geometry topics: matching two images through the feature cache, laying
// images side by side, and turning keypoints, matches and epipolar lines into overlays.

#include <deque>
#include <mutex>
#include <stop_token>
#include <vector>

#include <opencv2/core.hpp>

#include "app/module.hpp"
#include "core/geometry/homography.hpp"
#include "core/geometry/matching.hpp"
#include "core/geometry/sift.hpp"

namespace vl::app::geometry_common {

inline QPointF qp(const cv::Point2d& p) { return {p.x, p.y}; }

// SIFT features of the last few images, so that moving a slider that does not change the images
// (ratio test, RANSAC threshold...) skips detection, which is most of the run time. An entry keeps a
// reference to its image: while cached, that pixel buffer cannot be freed and reused by another image,
// so the buffer address identifies it. Thread-safe.
class FeatureCache {
public:
    // The features and the time their detection took (measured once, also reported on a cache hit).
    vl::geometry::Features get(const cv::Mat& image, const vl::geometry::SiftParams& p, std::stop_token stop,
                               double& ms);

private:
    struct Entry {
        cv::Mat image;  // shallow: shares the caller's pixels
        bool upsample = false;
        int maxFeatures = 0;
        vl::geometry::Features features;
        double ms = 0;  // what detection took
    };
    std::mutex mutex_;
    std::deque<Entry> entries_;
};

// SIFT on several images at once, one thread per image (the images are independent; the SIFT code itself
// stays sequential). ms[i] is the detection time of image i.
std::vector<vl::geometry::Features> detectAll(FeatureCache& cache, const std::vector<cv::Mat>& images,
                                              const vl::geometry::SiftParams& p, std::stop_token stop,
                                              std::vector<double>& ms);

// Putative correspondences between two images: matched keypoint positions, a[i] <-> b[i].
struct Correspondences {
    vl::geometry::Points a, b;
    double matchMs = 0;
};
Correspondences correspond(const vl::geometry::Features& fa, const vl::geometry::Features& fb,
                           const vl::geometry::MatchParams& mp, std::stop_token stop);

// The strongest keypoints as circles of radius 2σ with a tick along their orientation (at most `max`).
void addKeypoints(std::vector<Shape>& out, const std::vector<vl::geometry::Keypoint>& keypoints, cv::Point2d offset,
                  int max = 400);

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
