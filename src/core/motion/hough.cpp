// Generalised Hough transform tracker (see hough.hpp for the steps). Course: "3D Vision and Motion",
// part 10.

#include "core/motion/hough.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "core/common/filters.hpp"
#include "core/common/image.hpp"

namespace vl::motion {

// Step 1: gradient magnitude, orientation and its bin, and which pixels vote.
Orientations orientations(const cv::Mat& frame, const GradientParams& p) {
    CV_Assert(p.bins >= 2 && p.bins <= 1024);
    // Smooth first: the orientation of single-pixel noise is meaningless.
    const cv::Mat I = vl::gaussianBlur(toGrayF(frame), 1.0);
    const Gradient g = vl::sobel(I);
    const double pi = std::numbers::pi;
    Orientations o;
    o.magnitude.create(I.size(), CV_32F);
    o.angle.create(I.size(), CV_32F);
    o.bin.create(I.size(), CV_16U);
    o.mask.create(I.size(), CV_8U);
    for (int y = 0; y < I.rows; ++y)
        for (int x = 0; x < I.cols; ++x) {
            const float Ix = g.dx.at<float>(y, x), Iy = g.dy.at<float>(y, x);
            const float magnitude = std::sqrt(Ix * Ix + Iy * Iy);
            const float theta = std::atan2(Iy, Ix);
            // b = ⌊(θ + π) · bins / 2π⌋; θ = π would give `bins`, folded into the last bin.
            const int b = std::min(p.bins - 1, int((theta + pi) * p.bins / (2 * pi)));
            o.magnitude.at<float>(y, x) = magnitude;
            o.angle.at<float>(y, x) = theta;
            o.bin.at<ushort>(y, x) = ushort(b);
            o.mask.at<uchar>(y, x) = magnitude >= p.minMagnitude ? 255 : 0;
        }
    return o;
}

// Step 2: R(b) = { c − p : p in the box, p votes, b(p) = b }.
RTable buildRTable(const Orientations& o, cv::Rect box, int bins) {
    RTable t;
    t.bins = bins;
    t.offsets.assign(size_t(bins), {});
    t.centre = {box.x + box.width / 2, box.y + box.height / 2};
    const cv::Rect r = box & cv::Rect({0, 0}, o.mask.size());
    for (int y = r.y; y < r.br().y; ++y)
        for (int x = r.x; x < r.br().x; ++x) {
            if (!o.mask.at<uchar>(y, x)) continue;
            const int b = std::min<int>(o.bin.at<ushort>(y, x), bins - 1);
            t.offsets[size_t(b)].push_back(t.centre - cv::Point(x, y));
            ++t.entries;
        }
    return t;
}

namespace {

// The pixels q that can vote inside `region`: q + r ∈ region for some displacement r of the table,
// i.e. the region shifted by −r for the extreme r. Pixels outside it would only waste votes.
cv::Rect voterRegion(const RTable& t, cv::Rect region, cv::Rect frame) {
    int minX = 0, maxX = 0, minY = 0, maxY = 0;
    bool first = true;
    for (const auto& list : t.offsets)
        for (const cv::Point& r : list) {
            minX = first ? r.x : std::min(minX, r.x);
            maxX = first ? r.x : std::max(maxX, r.x);
            minY = first ? r.y : std::min(minY, r.y);
            maxY = first ? r.y : std::max(maxY, r.y);
            first = false;
        }
    const cv::Point topLeft(region.x - maxX, region.y - maxY), bottomRight(region.br().x - minX, region.br().y - minY);
    return cv::Rect(topLeft, bottomRight) & frame;
}

}  // namespace

// Step 3: every voting pixel q of bin b adds one vote at q + r for each r ∈ R(b).
cv::Mat houghVotes(const Orientations& o, const RTable& t, cv::Rect region) {
    const cv::Rect frame({0, 0}, o.mask.size());
    region = region.empty() ? frame : (region & frame);
    cv::Mat H = cv::Mat::zeros(o.mask.size(), CV_32F);
    if (region.empty() || t.entries == 0) return H;
    const cv::Rect voters = voterRegion(t, region, frame);
    for (int y = voters.y; y < voters.br().y; ++y)
        for (int x = voters.x; x < voters.br().x; ++x) {
            if (!o.mask.at<uchar>(y, x)) continue;
            const int b = std::min<int>(o.bin.at<ushort>(y, x), t.bins - 1);
            for (const cv::Point& r : t.offsets[size_t(b)]) {
                const cv::Point candidate(x + r.x, y + r.y);
                if (region.contains(candidate)) H.at<float>(candidate) += 1.f;
            }
        }
    return H;
}

namespace {

// The Kalman prior on the centre, as a Gaussian N(mu, S) with S = P⁻ + R (where the object should
// be, and how sure the filter is).
struct Prior {
    bool active = false;
    cv::Point2d mu;
    cv::Matx22d Sinv;
};

// Step 4: arg max over the region of H(x), or of H(x) · exp(−½ (x − μ)ᵀ S⁻¹ (x − μ)) with a prior:
// the vote map is read as a likelihood and the prior breaks ties between look-alikes.
cv::Point findPeak(const cv::Mat& H, cv::Rect region, const Prior& prior) {
    double best = -1;
    cv::Point arg(-1, -1);
    for (int y = region.y; y < region.br().y; ++y)
        for (int x = region.x; x < region.br().x; ++x) {
            const double votes = H.at<float>(y, x);
            if (votes <= 0) continue;
            double score = votes;
            if (prior.active) {
                const cv::Vec2d d(x - prior.mu.x, y - prior.mu.y);
                score *= std::exp(-0.5 * d.dot(prior.Sinv * d));
            }
            if (score > best) {
                best = score;
                arg = {x, y};
            }
        }
    return arg;
}

}  // namespace

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
    // 1. Orientations of the new frame.
    f.orient = orientations(frame, p_.gradient);

    // Where to look: around the last position, or the Kalman prediction.
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
    f.search = frameRect;
    if (p_.searchRadius > 0) {
        const int R = p_.searchRadius;
        const cv::Point c(int(std::lround(s.start.x - sub.x)), int(std::lround(s.start.y - sub.y)));
        f.search = cv::Rect(c.x - R, c.y - R, 2 * R + 1, 2 * R + 1) & frameRect;
    }

    // 3. Votes, restricted to the search region.
    f.votes = houghVotes(f.orient, table_, f.search);

    // 4. The peak, weighed by the prior N(prediction, P⁻ + R) when there is one.
    Prior prior;
    if (s.predicted) {
        const double r = p_.kalman.measurementNoise * p_.kalman.measurementNoise;
        prior = {true, s.prediction - sub, (s.predictionCov + cv::Matx22d(r, 0, 0, r)).inv()};
    }
    const cv::Point arg = findPeak(f.votes, f.search, prior);
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
        centre_ = detection;  // plain tracking by detection: the arg max, even when weak
    }
    s.box = boxAround(centre_, size_);
    return f;
}

}  // namespace vl::motion
