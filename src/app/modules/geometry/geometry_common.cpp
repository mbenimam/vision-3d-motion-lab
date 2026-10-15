#include "app/modules/geometry/geometry_common.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>

#include "core/common/image.hpp"
#include "core/common/task.hpp"
#include "core/geometry/fundamental.hpp"

namespace vl::app::geometry_common {

vl::geometry::Features FeatureCache::get(const cv::Mat& image, const vl::geometry::SiftParams& p,
                                         std::stop_token stop, double& ms) {
    auto same = [&](const Entry& e) {
        return e.image.data == image.data && e.image.size() == image.size() && e.upsample == p.upsample &&
               e.maxFeatures == p.maxFeatures;
    };
    {
        std::lock_guard lock(mutex_);
        for (const auto& e : entries_)
            if (same(e)) {
                ms = e.ms;
                return e.features;
            }
    }
    // Outside the lock: two runs may detect in parallel.
    const Stopwatch sw;
    vl::geometry::Features f = vl::geometry::detectSift(image, p, stop);
    ms = sw.ms();
    std::lock_guard lock(mutex_);
    entries_.push_front({image, p.upsample, p.maxFeatures, f, ms});
    if (entries_.size() > 6) entries_.pop_back();
    return f;
}

std::vector<vl::geometry::Features> detectAll(FeatureCache& cache, const std::vector<cv::Mat>& images,
                                              const vl::geometry::SiftParams& p, std::stop_token stop,
                                              std::vector<double>& ms) {
    ms.assign(images.size(), 0.0);
    std::vector<std::future<vl::geometry::Features>> jobs;
    for (size_t i = 0; i < images.size(); ++i)
        jobs.push_back(std::async(std::launch::async, [&, i] { return cache.get(images[i], p, stop, ms[i]); }));
    std::vector<vl::geometry::Features> out;
    for (auto& j : jobs) out.push_back(j.get());  // rethrows Cancelled
    return out;
}

Correspondences correspond(const vl::geometry::Features& fa, const vl::geometry::Features& fb,
                           const vl::geometry::MatchParams& mp, std::stop_token stop) {
    const Stopwatch sw;
    Correspondences c;
    for (const auto& m : vl::geometry::matchDescriptors(fa.descriptors, fb.descriptors, mp, stop)) {
        c.a.push_back(fa.keypoints[size_t(m.a)].pt);
        c.b.push_back(fb.keypoints[size_t(m.b)].pt);
    }
    c.matchMs = sw.ms();
    return c;
}

void addKeypoints(std::vector<Shape>& out, const std::vector<vl::geometry::Keypoint>& keypoints, cv::Point2d offset,
                  int max) {
    // Keypoints come sorted by contrast only when capped; pick the strongest explicitly.
    std::vector<size_t> order(keypoints.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return keypoints[a].contrast > keypoints[b].contrast; });
    if (order.size() > size_t(max)) order.resize(size_t(max));
    QColor c = colors::accent;
    c.setAlpha(200);
    for (size_t i : order) {
        const auto& k = keypoints[i];
        const QPointF centre = qp(k.pt + offset);
        const double r = 2 * k.sigma;
        out.push_back(EllipseShape{centre, r, r, 0, {c, 1.0}});
        out.push_back(LineShape{QLineF(centre, centre + QPointF(r * std::cos(k.angle), r * std::sin(k.angle))), {c, 1.0}});
    }
}

Strip sideBySide(const std::vector<cv::Mat>& images, int gap) {
    int W = 0, H = 0;
    for (const auto& im : images) W += im.cols, H = std::max(H, im.rows);
    W += gap * std::max(0, int(images.size()) - 1);
    Strip s;
    s.image = cv::Mat(H, W, CV_8UC3, kBackground);
    int x = 0;
    for (const auto& im : images) {
        vl::toBGR(im).copyTo(s.image(cv::Rect(x, 0, im.cols, im.rows)));
        s.offsets.emplace_back(x, 0);
        x += im.cols + gap;
    }
    return s;
}

void addMatchLines(std::vector<Shape>& out, const vl::geometry::Points& a, const vl::geometry::Points& b,
                   cv::Point2d offA, cv::Point2d offB, const std::vector<std::uint8_t>& inliers, bool showOutliers,
                   int maxOutliers, int maxInliers) {
    const size_t n = a.size();
    if (showOutliers) {
        size_t nOut = 0;
        for (size_t i = 0; i < n; ++i) nOut += !inliers[i];
        const double step = std::max(1.0, double(nOut) / maxOutliers);
        QColor c = colors::bad;
        c.setAlpha(150);
        double next = 0;
        size_t k = 0;
        for (size_t i = 0; i < n; ++i) {
            if (inliers[i]) continue;
            if (double(k++) < next) continue;
            next += step;
            out.push_back(LineShape{QLineF(qp(a[i] + offA), qp(b[i] + offB)), {c, 1.0}});
        }
    }
    // Inliers: at most maxInliers lines (evenly subsampled), translucent so dense bundles stay readable.
    size_t nIn = 0;
    for (size_t i = 0; i < n; ++i) nIn += inliers[i] != 0;
    const double stepIn = std::max(1.0, double(nIn) / maxInliers);
    QColor g = colors::good;
    g.setAlpha(nIn > size_t(maxInliers) ? 170 : 230);
    QPolygonF ends;
    double next = 0;
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!inliers[i]) continue;
        if (double(k++) < next) continue;
        next += stepIn;
        out.push_back(LineShape{QLineF(qp(a[i] + offA), qp(b[i] + offB)), {g, 1.0}});
        ends << qp(a[i] + offA) << qp(b[i] + offB);
    }
    out.push_back(PointsShape{ends, colors::good, 2.0});
}

std::optional<LineShape> lineShape(const cv::Vec3d& line, cv::Size size, cv::Point2d offset, Pen pen) {
    const auto seg = vl::geometry::clipLine(line, size);
    if (!seg) return std::nullopt;
    return LineShape{QLineF(qp(seg->first + offset), qp(seg->second + offset)), pen};
}

std::vector<int> spreadSubset(const vl::geometry::Points& pts, const std::vector<std::uint8_t>& mask, int n) {
    std::vector<int> cand;
    for (int i = 0; i < static_cast<int>(pts.size()); ++i)
        if (mask[size_t(i)]) cand.push_back(i);
    std::vector<int> out;
    if (cand.empty()) return out;
    // Start from the candidate nearest the centroid, then repeatedly take the farthest from those chosen.
    cv::Point2d c(0, 0);
    for (int i : cand) c += pts[size_t(i)];
    c *= 1.0 / double(cand.size());
    std::vector<double> dist(cand.size(), std::numeric_limits<double>::infinity());
    size_t pick = 0;
    for (size_t k = 0; k < cand.size(); ++k)
        if (cv::norm(pts[size_t(cand[k])] - c) < cv::norm(pts[size_t(cand[pick])] - c)) pick = k;
    while (static_cast<int>(out.size()) < std::min<int>(n, int(cand.size()))) {
        out.push_back(cand[pick]);
        size_t far = 0;
        for (size_t k = 0; k < cand.size(); ++k) {
            dist[k] = std::min(dist[k], cv::norm(pts[size_t(cand[k])] - pts[size_t(cand[pick])]));
            if (dist[k] > dist[far]) far = k;
        }
        pick = far;
    }
    return out;
}

QColor pairColor(int i) {
    // Golden-angle hue steps: neighbours in the list get very different hues.
    const double h = std::fmod(0.11 + i * 0.381966, 1.0);
    return QColor::fromHsvF(float(h), 0.62f, 1.0f);
}

}  // namespace vl::app::geometry_common
