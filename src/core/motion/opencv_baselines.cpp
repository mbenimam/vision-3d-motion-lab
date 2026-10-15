#include "core/motion/opencv_baselines.hpp"

#include <algorithm>
#include <cmath>

#include "core/common/task.hpp"

namespace vl::motion::opencv {
namespace {

cv::Mat grey(const cv::Mat& img) {
    if (img.channels() == 1) return img;
    cv::Mat g;
    cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    return g;
}

cv::Rect2d boxAround(cv::Point2d c, cv::Size2d s) { return {c.x - 0.5 * s.width, c.y - 0.5 * s.height, s.width, s.height}; }

}  // namespace

// ------------------------------------------------------------------ background subtraction

BackgroundSubtractor::BackgroundSubtractor(Subtractor which, const cv::Mat& background) {
    if (which == Subtractor::MOG2)
        impl_ = cv::createBackgroundSubtractorMOG2(500, 16.0, false);
    else
        impl_ = cv::createBackgroundSubtractorKNN(500, 400.0, false);
    if (!background.empty()) {
        cv::Mat ignored;
        impl_->apply(background, ignored, 1.0);
    }
}

BackgroundSubtractor::Step BackgroundSubtractor::apply(const cv::Mat& frame) {
    Step s;
    const Stopwatch sw;
    impl_->apply(frame, s.foreground);
    s.ms = sw.ms();
    s.foreground = s.foreground > 127;  // 255 foreground (shadows, if any, would be 127)
    return s;
}

// ------------------------------------------------------------------ mean shift / CamShift

ShiftTracker::ShiftTracker(Shift which, cv::Rect2d box, int maxIterations, double epsilon)
    : which_(which),
      window_(int(std::lround(box.x)), int(std::lround(box.y)), int(std::lround(box.width)), int(std::lround(box.height))),
      // OpenCV moves an integer window and compares the squared shift with round(ε²): an ε below
      // 1 px rounds to 0 and the loop would never stop early. Whole pixels it is.
      criteria_(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, maxIterations, std::max(1.0, epsilon)) {}

ShiftTracker::Step ShiftTracker::step(const cv::Mat& weights) {
    CV_Assert(weights.type() == CV_32F);
    cv::Mat w8;
    weights.convertTo(w8, CV_8U, 255.0);
    Step s;
    const Stopwatch sw;
    if (which_ == Shift::MeanShift)
        s.iterations = cv::meanShift(w8, window_, criteria_);
    else
        s.fitted = cv::CamShift(w8, window_, criteria_);
    s.ms = sw.ms();
    s.box = cv::Rect2d(window_);
    return s;
}

// ------------------------------------------------------------------ generalised Hough

BallardTracker::BallardTracker(const cv::Mat& firstFrame, cv::Rect2d box, const BallardParams& p)
    : ght_(cv::createGeneralizedHoughBallard()), size_(box.size()), centre_(box.x + 0.5 * box.width, box.y + 0.5 * box.height),
      radius_(p.searchRadius), lostRatio_(p.lostRatio) {
    // Settings set before the template: setTemplate computes the template's edges with them.
    ght_->setCannyLowThresh(std::max(1, int(std::lround(8 * p.minMagnitude))));
    ght_->setCannyHighThresh(std::max(2, int(std::lround(16 * p.minMagnitude))));
    ght_->setLevels(p.levels);
    ght_->setVotesThreshold(5);  // keep weak candidates too: we take the best one ourselves
    ght_->setMinDist(1.0);
    ght_->setDp(1.0);
    const cv::Rect b = cv::Rect(box) & cv::Rect({0, 0}, firstFrame.size());
    ght_->setTemplate(grey(firstFrame)(b).clone());  // reference point: the template's centre
}

BallardTracker::Step BallardTracker::step(const cv::Mat& frame) {
    Step s;
    const cv::Rect frameRect({0, 0}, frame.size());
    s.search = frameRect;
    if (radius_ > 0) {
        // Every centre within the radius, plus half the template around it.
        const int mx = radius_ + int(std::ceil(size_.width / 2)) + 2, my = radius_ + int(std::ceil(size_.height / 2)) + 2;
        const cv::Point c(int(std::lround(centre_.x)), int(std::lround(centre_.y)));
        s.search = cv::Rect(c.x - mx, c.y - my, 2 * mx + 1, 2 * my + 1) & frameRect;
    }
    const cv::Mat image = grey(frame)(s.search).clone();
    std::vector<cv::Vec4f> positions;
    std::vector<cv::Vec3i> votes;
    const Stopwatch sw;
    if (image.cols > 8 && image.rows > 8) ght_->detect(image, positions, votes);
    s.ms = sw.ms();

    // The detection with the most votes (OpenCV may return several local maxima).
    int best = -1;
    for (size_t i = 0; i < positions.size(); ++i)
        if (best < 0 || votes[i][0] > votes[size_t(best)][0]) best = int(i);
    s.votes = best >= 0 ? votes[size_t(best)][0] : 0.0;
    // Same "lost" rule as ours: a sudden drop against the recent typical peak.
    if (typicalVotes_ <= 0) typicalVotes_ = s.votes;
    s.lost = best < 0 || s.votes < lostRatio_ * typicalVotes_;
    if (!s.lost) typicalVotes_ = 0.9 * typicalVotes_ + 0.1 * s.votes;
    if (best >= 0) {
        const cv::Vec4f& found = positions[size_t(best)];  // (x, y, scale, angle) in the crop
        centre_ = {double(found[0]) + s.search.x, double(found[1]) + s.search.y};
    }
    s.box = boxAround(centre_, size_);
    return s;
}

}  // namespace vl::motion::opencv
