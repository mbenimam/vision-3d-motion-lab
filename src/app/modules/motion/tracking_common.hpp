#pragma once
// Pieces shared by the two tracking topics: the start box, the overlays on the frame and the summary.

#include <vector>

#include <QString>

#include <opencv2/core.hpp>

#include "app/module.hpp"
#include "core/motion/tracking.hpp"

namespace vl::app::tracking {

// The box to track from: the user's ROI if drawn (clipped to the frame), else motion_samples::defaultBox.
// note receives a hint when the box is only a guess.
cv::Rect2d startBox(const Inputs& in, const Params& p, QString* note);

// Box, trajectory so far, ground truth and the Kalman prediction ellipse of frame t.
std::vector<Shape> trackShapes(const std::vector<vl::motion::TrackStep>& steps, const cv::Rect2d& truth);

// The box and (when used) the prediction ellipse only: for the mechanism views.
std::vector<Shape> boxShapes(const vl::motion::TrackStep& s);

QRectF toQ(const cv::Rect2d& r);

// Per-run numbers shared by both trackers. score: the tracker's confidence per frame (%).
struct Summary {
    std::vector<double> iou, score, iterations;
    int lost = 0;
    bool hasTruth = false;
};
void summarise(Output& out, const Summary& s, const QString& scoreLabel, double ms, bool iterations);

}  // namespace vl::app::tracking
