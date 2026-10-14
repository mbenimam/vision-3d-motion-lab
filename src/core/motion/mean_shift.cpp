#include "core/motion/mean_shift.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace vl::motion {

HueImage hueOf(const cv::Mat& bgr, const HueParams& p) {
    CV_Assert(bgr.type() == CV_8UC3);
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    HueImage h;
    h.hue.create(bgr.size(), CV_8U);
    h.valid.create(bgr.size(), CV_8U);
    for (int y = 0; y < hsv.rows; ++y) {
        const cv::Vec3b* s = hsv.ptr<cv::Vec3b>(y);
        uchar* hu = h.hue.ptr<uchar>(y);
        uchar* v = h.valid.ptr<uchar>(y);
        for (int x = 0; x < hsv.cols; ++x) {
            hu[x] = s[x][0];
            v[x] = s[x][1] >= p.minSaturation && s[x][2] >= p.minValue ? 255 : 0;
        }
    }
    return h;
}

namespace {

inline int binOf(uchar hue, int bins) { return std::min(bins - 1, int(hue) * bins / 180); }

cv::Rect clipped(cv::Rect r, cv::Size s) { return r & cv::Rect({0, 0}, s); }

}  // namespace

std::vector<float> hueHistogram(const HueImage& h, cv::Rect box, int bins) {
    CV_Assert(bins >= 1 && bins <= 180);
    std::vector<float> hist(size_t(bins), 0.f);
    const cv::Rect r = clipped(box, h.hue.size());
    double total = 0;
    for (int y = r.y; y < r.y + r.height; ++y) {
        const uchar* hu = h.hue.ptr<uchar>(y);
        const uchar* v = h.valid.ptr<uchar>(y);
        for (int x = r.x; x < r.x + r.width; ++x)
            if (v[x]) {
                hist[size_t(binOf(hu[x], bins))] += 1.f;
                total += 1;
            }
    }
    if (total > 0)
        for (float& f : hist) f = float(f / total);
    return hist;
}

cv::Mat backProject(const HueImage& h, const std::vector<float>& hist) {
    const int bins = int(hist.size());
    const float peak = *std::max_element(hist.begin(), hist.end());
    // Lookup table over the 180 hues: one multiply-free read per pixel.
    float lut[180];
    for (int v = 0; v < 180; ++v) lut[v] = peak > 0 ? hist[size_t(binOf(uchar(v), bins))] / peak : 0.f;
    cv::Mat w(h.hue.size(), CV_32F);
    for (int y = 0; y < w.rows; ++y) {
        const uchar* hu = h.hue.ptr<uchar>(y);
        const uchar* v = h.valid.ptr<uchar>(y);
        float* o = w.ptr<float>(y);
        for (int x = 0; x < w.cols; ++x) o[x] = v[x] ? lut[std::min<int>(hu[x], 179)] : 0.f;
    }
    return w;
}

MeanShiftResult meanShift(const cv::Mat& weights, cv::Point2d start, cv::Size2d window, int maxIterations,
                          double epsilon) {
    CV_Assert(weights.type() == CV_32F && window.width >= 1 && window.height >= 1);
    MeanShiftResult res;
    res.centre = start;
    res.path.push_back(start);
    const int ww = int(std::lround(window.width)), wh = int(std::lround(window.height));
    for (int it = 0; it < maxIterations; ++it) {
        const cv::Rect win(int(std::lround(res.centre.x - 0.5 * window.width)),
                           int(std::lround(res.centre.y - 0.5 * window.height)), ww, wh);
        const cv::Rect r = clipped(win, weights.size());
        // Weighted centroid of the pixel centres (x + ½, y + ½) inside the window.
        double m = 0, mx = 0, my = 0;
        for (int y = r.y; y < r.y + r.height; ++y) {
            const float* w = weights.ptr<float>(y);
            double row = 0, rowx = 0;
            for (int x = r.x; x < r.x + r.width; ++x) {
                row += w[x];
                rowx += w[x] * (x + 0.5);
            }
            m += row;
            mx += rowx;
            my += row * (y + 0.5);
        }
        res.mass = m;
        if (m <= 1e-9) break;  // nothing of the model here: no direction to climb
        const cv::Point2d next(mx / m, my / m);
        ++res.iterations;
        const double shift = std::hypot(next.x - res.centre.x, next.y - res.centre.y);
        res.centre = next;
        res.path.push_back(next);
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
    first_.hue = hueOf(firstFrame, p_.hue);
    hist_ = hueHistogram(first_.hue, cv::Rect(box), p_.hue.bins);
    first_.weights = backProject(first_.hue, hist_);
    refMass_ = cv::sum(first_.weights(clipped(cv::Rect(box), firstFrame.size())))[0];
    first_.step.box = box;
    first_.step.start = centre_;
    first_.step.path = {centre_};
    first_.step.score = 1.0;
    kf_.init(centre_);
}

MeanShiftTracker::Frame MeanShiftTracker::step(const cv::Mat& frame) {
    Frame f;
    f.hue = hueOf(frame, p_.hue);
    f.weights = backProject(f.hue, hist_);
    TrackStep& s = f.step;
    s.start = centre_;
    if (p_.usePrediction) {
        kf_.predict();
        s.predicted = true;
        s.prediction = kf_.position();
        s.predictionCov = kf_.positionCovariance();
        s.start = s.prediction;
    }
    const MeanShiftResult ms = meanShift(f.weights, s.start, size_, p_.maxIterations, p_.epsilon);
    s.path = ms.path;
    s.iterations = ms.iterations;
    s.score = refMass_ > 0 ? std::min(1.0, ms.mass / refMass_) : 0.0;
    s.lost = s.score < p_.lostRatio;
    if (p_.usePrediction) {
        // A window holding little of the model is a weak (and, under partial occlusion, biased)
        // measurement: inflate R by 1/score², so the filter leans on its velocity instead.
        if (!s.lost) kf_.update(ms.centre, 1.0 / std::max(s.score * s.score, 1e-3));
        centre_ = kf_.position();
    } else {
        centre_ = ms.centre;
    }
    s.box = boxAround(centre_, size_);
    if (!s.lost && p_.modelUpdate > 0) {
        const auto now = hueHistogram(f.hue, cv::Rect(s.box), p_.hue.bins);
        const float g = float(p_.modelUpdate);
        for (size_t i = 0; i < hist_.size(); ++i) hist_[i] = (1.f - g) * hist_[i] + g * now[i];
    }
    return f;
}

}  // namespace vl::motion
