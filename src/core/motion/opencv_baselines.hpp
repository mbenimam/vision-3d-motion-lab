#pragma once
// OpenCV's own background subtraction and trackers, used as the baselines our implementations are
// compared to. Nothing here is part of the methods being taught: these are thin wrappers that run
// OpenCV on the same frames (and, for mean shift, on the same weight image) and return its answer in
// our conventions, with its time.

#include "core/common/cv_compat.hpp"  // OpenCV 4/5 headers: imgproc (GeneralizedHoughBallard), video

namespace vl::motion::opencv {

// ------------------------------------------------------------------ background subtraction
enum class Subtractor {
    MOG2,  // cv::BackgroundSubtractorMOG2: a mixture of Gaussians per pixel (Zivkovic 2004)
    KNN,   // cv::BackgroundSubtractorKNN: nearest neighbours among stored samples (Zivkovic 2006)
};

class BackgroundSubtractor {
public:
    // background: when not empty, the model starts from this picture (learning rate 1 on it), like
    // ours starting from the temporal median. Shadow detection is off: we have no shadow label.
    BackgroundSubtractor(Subtractor which, const cv::Mat& background);

    struct Step {
        cv::Mat foreground;  // CV_8U, 255 = foreground
        double ms = 0;
    };
    Step apply(const cv::Mat& frame);

private:
    cv::Ptr<cv::BackgroundSubtractor> impl_;
};

// ------------------------------------------------------------------ mean shift / CamShift
enum class Shift {
    MeanShift,  // cv::meanShift: fixed-size window
    CamShift,   // cv::CamShift: mean shift, then the window adapts its size and orientation
};

class ShiftTracker {
public:
    // epsilon is raised to 1 px: OpenCV's window moves by whole pixels.
    ShiftTracker(Shift which, cv::Rect2d box, int maxIterations, double epsilon);

    struct Step {
        cv::Rect2d box;         // the window after this frame
        cv::RotatedRect fitted; // CamShift's oriented box (empty for cv::meanShift)
        int iterations = -1;    // cv::meanShift's count; CamShift does not report it
        double ms = 0;
    };
    // weights: CV_32F in [0, 1], the same back-projection our tracker climbs (quantised to 8 bits,
    // the type OpenCV accepts).
    Step step(const cv::Mat& weights);

private:
    Shift which_;
    cv::Rect window_;
    cv::TermCriteria criteria_;
};

// ------------------------------------------------------------------ generalised Hough
// cv::GeneralizedHoughBallard with the first box as template, detecting in every frame on its own
// Canny edges and gradients. With searchRadius > 0 it only looks at a crop around the previous centre
// (the crop covers every centre within the radius plus the template around it), as ours does.
struct BallardParams {
    int levels = 36;             // R-table orientation bins (ours: GradientParams::bins)
    float minMagnitude = 4.f;    // our gradient threshold, in grey levels / px; Canny gets 8× and 16× it
                                 // (OpenCV's Sobel is not divided by 8 as ours is)
    int searchRadius = 0;        // 0 = whole frame
    double lostRatio = 0.4;
};

class BallardTracker {
public:
    BallardTracker(const cv::Mat& firstFrame, cv::Rect2d box, const BallardParams& p);

    struct Step {
        cv::Rect2d box;
        double votes = 0;  // OpenCV's votes at the detection (0 when nothing was found)
        bool lost = false; // no detection, or votes below lostRatio × the recent typical votes
        cv::Rect search;   // the part of the frame given to OpenCV
        double ms = 0;
    };
    Step step(const cv::Mat& frame);

private:
    cv::Ptr<cv::GeneralizedHoughBallard> ght_;
    cv::Size2d size_;
    cv::Point2d centre_;
    int radius_ = 0;
    double lostRatio_ = 0.4, typicalVotes_ = 0;
};

}  // namespace vl::motion::opencv
