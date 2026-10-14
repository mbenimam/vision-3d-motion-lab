// Mean-shift tracking on the back-projection of a hue histogram (see mean_shift.hpp for the steps).
// Course: "3D Vision and Motion", part 10.

#include "core/motion/mean_shift.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

#include "core/common/task.hpp"

namespace vl::motion {
namespace {

// Hue bin of an OpenCV hue in [0, 180).
int binOf(int hue, int bins) { return std::min(bins - 1, hue * bins / 180); }

cv::Rect clipped(cv::Rect r, cv::Size s) { return r & cv::Rect({0, 0}, s); }

}  // namespace

// Step 1. OpenCV's 8-bit HSV: H in [0, 180) (2° per unit), S and V in [0, 255].
HueImage hueOf(const cv::Mat& bgr, const HueParams& p) {
    CV_Assert(bgr.type() == CV_8UC3);
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    HueImage h;
    h.hue.create(bgr.size(), CV_8U);
    h.valid.create(bgr.size(), CV_8U);
    for (int y = 0; y < hsv.rows; ++y)
        for (int x = 0; x < hsv.cols; ++x) {
            const cv::Vec3b HSV = hsv.at<cv::Vec3b>(y, x);
            h.hue.at<uchar>(y, x) = HSV[0];
            h.valid.at<uchar>(y, x) = HSV[1] >= p.minSaturation && HSV[2] >= p.minValue ? 255 : 0;
        }
    return h;
}

// Step 2:  f_H(b) = #{s in box, valid, b(H(s)) = b} / #{s in box, valid}
std::vector<float> hueHistogram(const HueImage& h, cv::Rect box, int bins) {
    CV_Assert(bins >= 1 && bins <= 180);
    std::vector<float> f(size_t(bins), 0.f);
    const cv::Rect r = clipped(box, h.hue.size());
    double count = 0;
    for (int y = r.y; y < r.br().y; ++y)
        for (int x = r.x; x < r.br().x; ++x)
            if (h.valid.at<uchar>(y, x)) {
                f[size_t(binOf(h.hue.at<uchar>(y, x), bins))] += 1.f;
                count += 1;
            }
    if (count > 0)
        for (float& v : f) v = float(v / count);
    return f;
}

// Step 3:  R_H(x, y) = f_H(b(H(x, y))) / max_b f_H(b)
cv::Mat backProject(const HueImage& h, const std::vector<float>& f) {
    const int bins = int(f.size());
    const float peak = *std::max_element(f.begin(), f.end());
    cv::Mat R(h.hue.size(), CV_32F);
    for (int y = 0; y < R.rows; ++y)
        for (int x = 0; x < R.cols; ++x) {
            const bool ok = h.valid.at<uchar>(y, x) && peak > 0;
            R.at<float>(y, x) = ok ? f[size_t(binOf(h.hue.at<uchar>(y, x), bins))] / peak : 0.f;
        }
    return R;
}

double windowMass(const cv::Mat& weights, cv::Rect box) {
    const cv::Rect r = clipped(box, weights.size());
    double m = 0;
    for (int y = r.y; y < r.br().y; ++y)
        for (int x = r.x; x < r.br().x; ++x) m += weights.at<float>(y, x);
    return m;
}

namespace {

// The weighted centroid of the pixel centres (x + ½, y + ½) inside the window centred on c.
struct Centroid {
    cv::Point2d centre;
    double mass = 0;
};

Centroid weightedCentroid(const cv::Mat& w, cv::Point2d c, cv::Size2d window) {
    const cv::Rect W(int(std::lround(c.x - 0.5 * window.width)), int(std::lround(c.y - 0.5 * window.height)),
                     int(std::lround(window.width)), int(std::lround(window.height)));
    const cv::Rect r = clipped(W, w.size());
    double m = 0, mx = 0, my = 0;
    for (int y = r.y; y < r.br().y; ++y)
        for (int x = r.x; x < r.br().x; ++x) {
            const double v = w.at<float>(y, x);
            m += v;
            mx += v * (x + 0.5);
            my += v * (y + 0.5);
        }
    return {m > 0 ? cv::Point2d(mx / m, my / m) : c, m};
}

}  // namespace

// Step 4:  c_{k+1} = Σ R_H(s) · s / Σ R_H(s) over the window at c_k, until |c_{k+1} − c_k| < ε.
MeanShiftResult meanShift(const cv::Mat& weights, cv::Point2d start, cv::Size2d window, int maxIterations,
                          double epsilon) {
    CV_Assert(weights.type() == CV_32F && window.width >= 1 && window.height >= 1);
    MeanShiftResult res;
    res.centre = start;
    res.path.push_back(start);
    for (int k = 0; k < maxIterations; ++k) {
        const Centroid next = weightedCentroid(weights, res.centre, window);
        res.mass = next.mass;
        if (next.mass <= 1e-9) break;  // nothing of the model here: no direction to climb
        const double shift = std::hypot(next.centre.x - res.centre.x, next.centre.y - res.centre.y);
        res.centre = next.centre;
        res.path.push_back(next.centre);
        ++res.iterations;
        if (shift < epsilon) {
            res.converged = true;
            break;
        }
    }
    return res;
}

MeanShiftTracker::MeanShiftTracker(const MeanShiftTrackerParams& p, const cv::Mat& firstFrame, cv::Rect2d box)
    : p_(p), size_(box.size()), centre_(centreOf(box)), kf_(p.kalman) {
    CV_Assert(box.width >= 2 && box.height >= 2);
    // The model is learnt once, on the first box.
    first_.hue = hueOf(firstFrame, p_.hue);
    hist_ = hueHistogram(first_.hue, cv::Rect(box), p_.hue.bins);
    first_.weights = backProject(first_.hue, hist_);
    refMass_ = windowMass(first_.weights, cv::Rect(box));
    first_.step.box = box;
    first_.step.start = centre_;
    first_.step.path = {centre_};
    first_.step.score = 1.0;
    kf_.init(centre_);
}

MeanShiftTracker::Frame MeanShiftTracker::step(const cv::Mat& frame) {
    Frame f;
    TrackStep& s = f.step;
    // Steps 1 and 3: the weights of this frame under the model.
    f.hue = hueOf(frame, p_.hue);
    f.weights = backProject(f.hue, hist_);

    // Where to start climbing: the last position, or the Kalman prediction.
    s.start = centre_;
    if (p_.usePrediction) {
        kf_.predict();
        s.predicted = true;
        s.prediction = kf_.position();
        s.predictionCov = kf_.positionCovariance();
        s.start = s.prediction;
    }

    // Step 4: climb.
    const Stopwatch clock;
    const MeanShiftResult ms = meanShift(f.weights, s.start, size_, p_.maxIterations, p_.epsilon);
    f.climbMs = clock.ms();
    s.path = ms.path;
    s.iterations = ms.iterations;
    s.score = refMass_ > 0 ? std::min(1.0, ms.mass / refMass_) : 0.0;
    s.lost = s.score < p_.lostRatio;

    // With the filter, mean shift's answer is a measurement. A window holding little of the model is
    // a weak (and, under partial occlusion, biased) one: R is inflated by 1/score², so the filter
    // leans on its velocity instead. When lost, only the prediction remains.
    if (p_.usePrediction) {
        if (!s.lost) kf_.update(ms.centre, 1.0 / std::max(s.score * s.score, 1e-3));
        centre_ = kf_.position();
    } else {
        centre_ = ms.centre;
    }
    s.box = boxAround(centre_, size_);

    // Optional slow model update: f_H ← (1 − γ) f_H + γ f_H(new box).
    if (!s.lost && p_.modelUpdate > 0) {
        const std::vector<float> now = hueHistogram(f.hue, cv::Rect(s.box), p_.hue.bins);
        const float g = float(p_.modelUpdate);
        for (size_t b = 0; b < hist_.size(); ++b) hist_[b] = (1.f - g) * hist_[b] + g * now[b];
    }
    return f;
}

}  // namespace vl::motion
