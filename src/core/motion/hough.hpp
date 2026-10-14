#pragma once
// Tracking by detection with the generalised Hough transform (Ballard 1981). Course: "3D Vision and
// Motion", part 10 (Hough transforms), and its tracking practical.
//
// The steps, one function each:
//   1. orientations   smooth the frame (σ = 1), take its gradient (Ix, Iy) with Sobel, then for
//                     every pixel the magnitude |∇I| = √(Ix² + Iy²), the orientation
//                     θ = atan2(Iy, Ix) and its bin b = ⌊(θ + π) · bins / 2π⌋. Only pixels with
//                     |∇I| above a threshold vote: the orientation of a flat area is noise.
//   2. buildRTable    the model, from the first box: for each orientation bin b, the list R(b) of
//                     displacements r = c − p from the box's voting pixels p of that bin to its
//                     centre c
//   3. houghVotes     in a new frame, each voting pixel q of bin b votes for every q + r, r ∈ R(b):
//                         H(x) = #{(q, r) : r ∈ R(b(q)), q + r = x}
//                     The object's edge pixels all vote for its centre, wherever it has moved.
//   4. findPeak       the detection is the arg max of H (weighed, with a prediction, by the Kalman
//                     prior N(prediction, P⁻ + R) so that a look-alike far away cannot win)
//
// Cost of step 3: Σ over voting pixels of the length of their bin's list, about
// (voting pixels) × (model pixels) / bins. Restricting the search to a window around the previous
// (or predicted) position means only the pixels that can reach the window vote.

#include <vector>

#include <opencv2/core.hpp>

#include "core/motion/kalman.hpp"
#include "core/motion/tracking.hpp"

namespace vl::motion {

struct GradientParams {
    int bins = 36;             // orientation bins over [−π, π)
    float minMagnitude = 4.f;  // |∇I| below this does not vote (grey levels per pixel)
};

struct Orientations {
    cv::Mat magnitude;  // CV_32F, |∇I| in grey levels / px
    cv::Mat angle;      // CV_32F, θ = atan2(Iy, Ix) in (−π, π]
    cv::Mat bin;        // CV_16U, orientation bin of θ
    cv::Mat mask;       // CV_8U, 255 where the pixel votes (|∇I| ≥ threshold)
};
Orientations orientations(const cv::Mat& frame, const GradientParams& p);

// The R-table: for each orientation bin, the displacements from the model's pixels of that
// orientation to the reference point (the box centre, rounded to a pixel).
struct RTable {
    int bins = 0;
    std::vector<std::vector<cv::Point>> offsets;  // offsets[b] = R(b), each r = centre − p
    long entries = 0;                             // total number of displacements
    cv::Point centre;                             // reference point c (integer pixel)
};
RTable buildRTable(const Orientations& o, cv::Rect box, int bins);

// Accumulator H(x) = number of votes for "the centre is at x" (CV_32F, frame size). region: only
// candidates inside it are counted (empty = the whole frame).
cv::Mat houghVotes(const Orientations& o, const RTable& t, cv::Rect region = {});

struct HoughTrackerParams {
    GradientParams gradient;
    int searchRadius = 0;        // > 0: only centres within this many px of the previous / predicted one
    bool usePrediction = false;  // weigh the votes by the Kalman prior N(prediction, P⁻ + R) before the arg max
    KalmanParams kalman;
    double lostRatio = 0.4;      // lost when the peak falls under this share of its recent typical value
};

class HoughTracker {
public:
    struct Frame {
        TrackStep step;
        Orientations orient;
        cv::Mat votes;       // the raw accumulator H
        cv::Rect search;     // region searched (the frame when global)
        double peak = 0;     // H at the detection
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
