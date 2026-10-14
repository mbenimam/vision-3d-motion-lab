#pragma once
// Mean-shift tracking on a colour model (Comaniciu, Ramesh & Meer). Course: "3D Vision and Motion",
// part 10 (tracking a distribution), and its tracking practical.
//
// The steps, one function each:
//   1. hueOf          the hue H(x, y) of every pixel, and whether it is reliable (saturation and
//                     value high enough: grey and dark pixels have an unstable hue)
//   2. hueHistogram   the model f_H(b) = share of the box's reliable pixels whose hue falls in bin b
//   3. backProject    the weight of each pixel, R_H(x, y) = f_H(b(H(x, y))) / max_b f_H(b): how
//                     typical of the object its colour is
//   4. meanShift      move the window centre c to the weighted centroid of R_H inside the window,
//                         c_{k+1} = Σ_{s ∈ W(c_k)} R_H(s) · s  /  Σ_{s ∈ W(c_k)} R_H(s),
//                     until it moves by less than ε. Each step climbs the density of model-coloured
//                     pixels, so the window settles on the nearest mode without an exhaustive search.
//
// Costs: steps 1 and 3 are O(W·H) per frame (3 is one table lookup per pixel); step 4 is
// O(iterations · window area). The tracker can start each search from a Kalman prediction and coast
// on it when the window holds too little of the model (an occlusion).

#include <vector>

#include <opencv2/core.hpp>

#include "core/motion/kalman.hpp"
#include "core/motion/tracking.hpp"

namespace vl::motion {

struct HueParams {
    int bins = 32;           // histogram bins over the 180 OpenCV hue values (2° each)
    int minSaturation = 60;  // pixels below either level have no reliable hue and carry no weight
    int minValue = 32;
};

struct HueImage {
    cv::Mat hue;    // CV_8U, OpenCV hue in [0, 180)
    cv::Mat valid;  // CV_8U, 255 where saturation and value pass the thresholds
};
HueImage hueOf(const cv::Mat& bgr, const HueParams& p);

// f_H: density of the valid hues inside box (sums to 1; all zero if no pixel is valid).
std::vector<float> hueHistogram(const HueImage& h, cv::Rect box, int bins);

// R_H(x, y) = f_H(H(x, y)) / max f_H, in [0, 1]; 0 on invalid pixels. CV_32F.
cv::Mat backProject(const HueImage& h, const std::vector<float>& hist);

// Σ of the weights inside box (clipped to the image): how much of the model the box holds.
double windowMass(const cv::Mat& weights, cv::Rect box);

struct MeanShiftResult {
    cv::Point2d centre;
    std::vector<cv::Point2d> path;  // start, then each new centre
    int iterations = 0;
    double mass = 0;                // Σ weights in the final window
    bool converged = false;
};

// Climb the weight image from start with a window of the given size. Stops when the centre moves by
// less than epsilon px, or after maxIterations; an empty window (mass 0) leaves the centre where it is.
MeanShiftResult meanShift(const cv::Mat& weights, cv::Point2d start, cv::Size2d window, int maxIterations = 20,
                          double epsilon = 0.5);

struct MeanShiftTrackerParams {
    HueParams hue;
    double modelUpdate = 0.0;    // γ: f_H ← (1 − γ) f_H + γ f_H(new box) after each confident frame
    bool usePrediction = false;  // start each search from a Kalman prediction, coast on it when lost
    KalmanParams kalman;
    double lostRatio = 0.3;      // lost when the window holds less than this share of the first box's mass
    int maxIterations = 20;
    double epsilon = 0.5;
};

class MeanShiftTracker {
public:
    struct Frame {
        TrackStep step;
        HueImage hue;
        cv::Mat weights;  // back-projection R_H used for this frame
        double climbMs = 0;  // time of step 4 alone (the part cv::meanShift does)
    };

    MeanShiftTracker(const MeanShiftTrackerParams& p, const cv::Mat& firstFrame, cv::Rect2d box);
    const Frame& first() const { return first_; }  // frame 0: the box as given
    Frame step(const cv::Mat& frame);
    const std::vector<float>& model() const { return hist_; }
    double referenceMass() const { return refMass_; }  // window mass of the first box

private:
    MeanShiftTrackerParams p_;
    std::vector<float> hist_;  // f_H
    cv::Size2d size_;
    cv::Point2d centre_;
    double refMass_ = 0;
    ConstantVelocityKalman kf_;
    Frame first_;
};

}  // namespace vl::motion
