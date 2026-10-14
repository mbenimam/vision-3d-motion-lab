#include "core/motion/hough.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <opencv2/imgproc.hpp>

#include "core/common/image.hpp"

namespace vl::motion {

Orientations orientations(const cv::Mat& frame, const GradientParams& p) {
    CV_Assert(p.bins >= 2 && p.bins <= 1024);
    cv::Mat g = toGrayF(frame);
    cv::GaussianBlur(g, g, {0, 0}, 1.0);  // orientation of single-pixel noise is meaningless
    cv::Mat gx, gy;
    cv::Sobel(g, gx, CV_32F, 1, 0, 3, 1.0 / 8.0);  // 1/8: the 3×3 Sobel weights sum to 8
    cv::Sobel(g, gy, CV_32F, 0, 1, 3, 1.0 / 8.0);
    Orientations o;
    o.magnitude.create(g.size(), CV_32F);
    o.angle.create(g.size(), CV_32F);
    o.bin.create(g.size(), CV_16U);
    o.mask.create(g.size(), CV_8U);
    const float scale = float(p.bins / (2 * std::numbers::pi));
    for (int y = 0; y < g.rows; ++y) {
        const float* dx = gx.ptr<float>(y);
        const float* dy = gy.ptr<float>(y);
        float* m = o.magnitude.ptr<float>(y);
        float* a = o.angle.ptr<float>(y);
        ushort* b = o.bin.ptr<ushort>(y);
        uchar* k = o.mask.ptr<uchar>(y);
        for (int x = 0; x < g.cols; ++x) {
            m[x] = std::sqrt(dx[x] * dx[x] + dy[x] * dy[x]);
            a[x] = std::atan2(dy[x], dx[x]);
            b[x] = ushort(std::min(p.bins - 1, int((a[x] + float(std::numbers::pi)) * scale)));
            k[x] = m[x] >= p.minMagnitude ? 255 : 0;
        }
    }
    return o;
}

RTable buildRTable(const Orientations& o, cv::Rect box, int bins) {
    RTable t;
    t.bins = bins;
    t.offsets.assign(size_t(bins), {});
    t.centre = {box.x + box.width / 2, box.y + box.height / 2};
    const cv::Rect r = box & cv::Rect({0, 0}, o.mask.size());
    for (int y = r.y; y < r.y + r.height; ++y) {
        const uchar* k = o.mask.ptr<uchar>(y);
        const ushort* b = o.bin.ptr<ushort>(y);
        for (int x = r.x; x < r.x + r.width; ++x)
            if (k[x]) {
                t.offsets[std::min<size_t>(b[x], size_t(bins - 1))].push_back(t.centre - cv::Point(x, y));
                ++t.entries;
            }
    }
    return t;
}

cv::Mat houghVotes(const Orientations& o, const RTable& t, cv::Rect region) {
    const cv::Rect frame({0, 0}, o.mask.size());
    region = region.empty() ? frame : (region & frame);
    cv::Mat acc = cv::Mat::zeros(o.mask.size(), CV_32F);
    if (region.empty() || t.entries == 0) return acc;
    // Only voters q with q + r inside the region for some r can contribute: the region shifted by −r.
    int minX = 0, maxX = 0, minY = 0, maxY = 0;
    bool any = false;
    for (const auto& list : t.offsets)
        for (const cv::Point& r : list) {
            minX = any ? std::min(minX, r.x) : r.x;
            maxX = any ? std::max(maxX, r.x) : r.x;
            minY = any ? std::min(minY, r.y) : r.y;
            maxY = any ? std::max(maxY, r.y) : r.y;
            any = true;
        }
    const cv::Rect voters = cv::Rect(cv::Point(region.x - maxX, region.y - maxY),
                                     cv::Point(region.br().x - minX, region.br().y - minY)) & frame;
    const int x0 = region.x, x1 = region.br().x, y0 = region.y, y1 = region.br().y;
    for (int y = voters.y; y < voters.br().y; ++y) {
        const uchar* k = o.mask.ptr<uchar>(y);
        const ushort* b = o.bin.ptr<ushort>(y);
        for (int x = voters.x; x < voters.br().x; ++x) {
            if (!k[x]) continue;
            for (const cv::Point& r : t.offsets[std::min<size_t>(b[x], t.offsets.size() - 1)]) {
                const int ax = x + r.x, ay = y + r.y;
                if (ax >= x0 && ax < x1 && ay >= y0 && ay < y1) acc.ptr<float>(ay)[ax] += 1.f;
            }
        }
    }
    return acc;
}

HoughTracker::HoughTracker(const HoughTrackerParams& p, const cv::Mat& firstFrame, cv::Rect2d box)
    : p_(p), size_(box.size()), centre_(centreOf(box)), kf_(p.kalman) {
    CV_Assert(box.width >= 2 && box.height >= 2);
    first_.orient = orientations(firstFrame, p_.gradient);
    table_ = buildRTable(first_.orient, cv::Rect(box), p_.gradient.bins);
    first_.votes = houghVotes(first_.orient, table_);
    first_.search = cv::Rect({0, 0}, firstFrame.size());
    first_.peak = double(table_.entries);
    first_.step.box = box;
    first_.step.start = centre_;
    first_.step.path = {centre_};
    first_.step.score = 1.0;
    kf_.init(centre_);
}

HoughTracker::Frame HoughTracker::step(const cv::Mat& frame) {
    Frame f;
    TrackStep& s = f.step;
    f.orient = orientations(frame, p_.gradient);
    s.start = centre_;
    if (p_.usePrediction) {
        kf_.predict();
        s.predicted = true;
        s.prediction = kf_.position();
        s.predictionCov = kf_.positionCovariance();
        s.start = s.prediction;
    }
    // The table votes for its integer reference point; the box centre sits `sub` away from it.
    const cv::Point2d sub = centreOf(first_.step.box) - cv::Point2d(table_.centre);
    const cv::Rect frameRect({0, 0}, frame.size());
    if (p_.searchRadius > 0) {
        const cv::Point c(int(std::lround(s.start.x - sub.x)), int(std::lround(s.start.y - sub.y)));
        const int R = p_.searchRadius;
        f.search = cv::Rect(c.x - R, c.y - R, 2 * R + 1, 2 * R + 1) & frameRect;
    } else {
        f.search = frameRect;
    }
    f.votes = houghVotes(f.orient, table_, f.search);

    // Arg max of the votes, or of votes × N(prediction, P + R) when a prediction is available: the
    // whole vote map is read as a likelihood and the Kalman prior breaks ties between look-alikes.
    cv::Matx22d Si;
    cv::Point2d mu;
    if (s.predicted) {
        const double r2 = p_.kalman.measurementNoise * p_.kalman.measurementNoise;
        Si = (s.predictionCov + cv::Matx22d(r2, 0, 0, r2)).inv();
        mu = s.prediction - sub;
    }
    double best = -1;
    cv::Point arg(-1, -1);
    for (int y = f.search.y; y < f.search.br().y; ++y) {
        const float* v = f.votes.ptr<float>(y);
        for (int x = f.search.x; x < f.search.br().x; ++x) {
            if (v[x] <= 0) continue;
            double score = v[x];
            if (s.predicted) {
                const double dx = x - mu.x, dy = y - mu.y;
                score *= std::exp(-0.5 * (dx * (Si(0, 0) * dx + Si(0, 1) * dy) + dy * (Si(1, 0) * dx + Si(1, 1) * dy)));
            }
            if (score > best) {
                best = score;
                arg = {x, y};
            }
        }
    }
    f.peak = arg.x >= 0 ? f.votes.at<float>(arg) : 0.0;
    s.score = table_.entries ? f.peak / double(table_.entries) : 0.0;
    // Real objects turn and deform, so the share of the model found drifts far below 1 while the
    // detection is still right; "lost" means a sudden drop against the recent typical peak.
    if (typicalPeak_ <= 0) typicalPeak_ = f.peak;
    s.lost = f.peak < p_.lostRatio * typicalPeak_;
    if (!s.lost) typicalPeak_ = 0.9 * typicalPeak_ + 0.1 * f.peak;
    const cv::Point2d detection = arg.x >= 0 ? cv::Point2d(arg) + sub : s.start;
    s.path = {s.start, detection};
    if (p_.usePrediction) {
        if (!s.lost) kf_.update(detection);
        centre_ = kf_.position();
    } else {
        centre_ = detection;  // straightforward tracking: the arg max, even when weak
    }
    s.box = boxAround(centre_, size_);
    return f;
}

}  // namespace vl::motion
