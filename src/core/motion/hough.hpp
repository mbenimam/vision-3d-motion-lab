#pragma once
// Tracking by detection with the generalised Hough transform (Ballard), as in the course practical.
//
// Each pixel gets its gradient orientation θ (Sobel) and magnitude; only pixels with a magnitude above
// a threshold vote, since the orientation of a flat area is noise. The model is an R-table: for each
// orientation bin, the list of displacements r = c − p from the box's edge pixels p to its centre c.
// In a new frame, a pixel q of orientation bin b votes for every q + r listed under b; the object's
// centre collects the votes of all its edge pixels at once, and the arg max of the accumulator is the
// detection. Cost: Σ over voting pixels of the length of their bin's list, about
// (voting pixels) × (model edge pixels) / bins per frame. Restricting the search to a window around
// the previous (or predicted) position only lets the pixels that can reach the window vote.

#include <vector>

#include <opencv2/core.hpp>

#include "core/motion/kalman.hpp"
#include "core/motion/tracking.hpp"

namespace vl::motion {

struct GradientParams {
    int bins = 36;               // orientation bins over [−π, π)
    float minMagnitude = 4.f;    // |∇I| below this does not vote (grey levels per pixel)
};

struct Orientations {
    cv::Mat magnitude;  // CV_32F, |∇I| in grey levels / px
    cv::Mat angle;      // CV_32F, θ = atan2(∂y I, ∂x I) in (−π, π]
    cv::Mat bin;        // CV_16U, orientation bin of θ
    cv::Mat mask;       // CV_8U, 255 where the pixel votes (magnitude ≥ threshold)
};
Orientations orientations(const cv::Mat& frame, const GradientParams& p);

struct RTable {
    int bins = 0;
    std::vector<std::vector<cv::Point>> offsets;  // per orientation bin: centre − p
    long entries = 0;
    cv::Point centre;                             // reference point used to build it (integer pixel)
};
RTable buildRTable(const Orientations& o, cv::Rect box, int bins);

// Accumulator H(x) = number of votes for "the centre is at x" (CV_32F, frame size). region: only
// candidates inside it are counted (empty = the whole frame).
cv::Mat houghVotes(const Orientations& o, const RTable& t, cv::Rect region = {});

struct HoughTrackerParams {
    GradientParams gradient;
    int searchRadius = 0;        // > 0: only centres within this many px of the previous / predicted one
    bool usePrediction = false;  // weigh the votes by the Kalman prior N(prediction, P) before the arg max
    KalmanParams kalman;
    double lostRatio = 0.4;      // lost when the peak falls under this share of its recent typical value
};

class HoughTracker {
public:
    struct Frame {
        TrackStep step;
        Orientations orient;
        cv::Mat votes;       // raw accumulator
        cv::Rect search;     // region searched (the frame when global)
        double peak = 0;     // votes at the detection
    };

    HoughTracker(const HoughTrackerParams& p, const cv::Mat& firstFrame, cv::Rect2d box);
    const Frame& first() const { return first_; }
    Frame step(const cv::Mat& frame);
    const RTable& table() const { return table_; }

private:
    HoughTrackerParams p_;
    RTable table_;
    cv::Size2d size_;
    cv::Point2d centre_;
    double typicalPeak_ = 0;  // running average of the confident peaks (frame 0 matches itself: excluded)
    ConstantVelocityKalman kf_;
    Frame first_;
};

}  // namespace vl::motion
