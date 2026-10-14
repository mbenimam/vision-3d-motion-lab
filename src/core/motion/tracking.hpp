#pragma once
// Pieces shared by the trackers (course "3D Vision and Motion", part 10): the per-frame result, box
// helpers and the overlap score IoU(a, b) = area(a ∩ b) / area(a ∪ b).

#include <vector>

#include <opencv2/core.hpp>

namespace vl::motion {

// One frame of a track. The box keeps the size of the first one (no scale change is modelled).
struct TrackStep {
    cv::Rect2d box;                   // estimate for this frame
    cv::Point2d start;                // where the search began (previous position or prediction)
    std::vector<cv::Point2d> path;    // successive centres of the search (mean shift), start first
    int iterations = 0;
    double score = 0;                 // confidence in [0, 1]: kept model mass (mean shift), vote share (Hough)
    bool lost = false;                // score below the tracker's threshold: the estimate is the prediction
    bool predicted = false;           // a Kalman prediction was used this frame
    cv::Point2d prediction;           // its position
    cv::Matx22d predictionCov;        // and its 2×2 position covariance (before the measurement)
};

inline cv::Point2d centreOf(const cv::Rect2d& r) { return {r.x + 0.5 * r.width, r.y + 0.5 * r.height}; }
inline cv::Rect2d boxAround(cv::Point2d c, cv::Size2d s) { return {c.x - 0.5 * s.width, c.y - 0.5 * s.height, s.width, s.height}; }

// Intersection over union of two boxes, in [0, 1].
double iou(const cv::Rect2d& a, const cv::Rect2d& b);

}  // namespace vl::motion
