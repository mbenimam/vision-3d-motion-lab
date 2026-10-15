// Choosing the page among the Hough lines, refining its sides and ordering its corners.
// Course "3D Vision and Motion": the scanner application (the corners feed the DLT of part 2).
//
//   1. quadFromLines      opposite sides (p1, p2) and (q1, q2) give the cycle p1, q1, p2, q2 and the
//                         corners p1∩q1, q1∩p2, p2∩q2, q2∩p1
//   2. isPlausible        a page seen in perspective is still a convex quadrilateral with no very
//                         sharp or very flat corner
//   3. sideSupport        S = (points of the side with an edge pixel of the same orientation within
//                         1 px) / (points of the side)
//   4. bestQuadrilateral  argmax over candidates of  area · S₁ S₂ S₃ S₄
//   5. refineSide         sub-pixel step positions across the side, then total least squares
//   6. orderCorners       angle around the centroid
//
// Why area × product of supports: the area alone would prefer a quadrilateral that runs past the
// page along the desk's edge; the supports alone would accept a small box made of two text lines.
// The product makes one weak side expensive (0.6 support costs 40 %), while an occluded corner
// (two sides each missing a little) still wins over any smaller box.

#include "core/scanner/quad.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "core/common/filters.hpp"

namespace vl::scanner {
namespace {

constexpr double kDegree = std::numbers::pi / 180.0;

// Step 1. Returns false when two adjacent sides are parallel.
bool quadFromLines(const HoughLine& p1, const HoughLine& q1, const HoughLine& p2, const HoughLine& q2, Quad& corners) {
    return intersect(p1, q1, corners[0]) && intersect(q1, p2, corners[1]) && intersect(p2, q2, corners[2]) &&
           intersect(q2, p1, corners[3]);
}

// z component of (b − a) × (c − b): its sign says whether the path a → b → c turns left or right.
double turn(cv::Point2d a, cv::Point2d b, cv::Point2d c) { return (b - a).cross(c - b); }

// Step 2. Convex: the four turns have the same sign. The interior angle at corner i is the angle
// between the vectors to its two neighbours.
bool isPlausible(const Quad& c, cv::Size image, const QuadParams& p) {
    const double mx = p.margin * image.width, my = p.margin * image.height;
    for (const auto& q : c)
        if (q.x < -mx || q.y < -my || q.x > image.width + mx || q.y > image.height + my) return false;
    int positive = 0;
    for (int i = 0; i < 4; ++i) positive += turn(c[size_t(i)], c[size_t((i + 1) % 4)], c[size_t((i + 2) % 4)]) > 0;
    if (positive != 0 && positive != 4) return false;
    for (int i = 0; i < 4; ++i) {
        const cv::Point2d u = c[size_t((i + 3) % 4)] - c[size_t(i)], v = c[size_t((i + 1) % 4)] - c[size_t(i)];
        const double angle = std::acos(std::clamp(u.dot(v) / (std::hypot(u.x, u.y) * std::hypot(v.x, v.y)), -1.0, 1.0));
        if (angle < p.minAngleDeg * kDegree || angle > p.maxAngleDeg * kDegree) return false;
    }
    return quadArea(c) >= p.minAreaShare * image.area();
}

// Step 3: walk from a to b one pixel at a time. A point counts when an edge pixel within 1 px has a
// gradient roughly perpendicular to the side (within 20°): a text line crossing the side does not.
double sideSupport(cv::Point2d a, cv::Point2d b, const EdgeMap& e) {
    const cv::Point2d d = b - a;
    const double length = std::hypot(d.x, d.y);
    if (length < 1) return 0;
    const cv::Point2d normal(-d.y / length, d.x / length);
    const double cosTolerance = std::cos(20 * kDegree);
    const int steps = int(length);
    int supported = 0, inside = 0;
    for (int s = 0; s <= steps; ++s) {
        const cv::Point2d q = a + d * (double(s) / steps);
        const int x = int(std::lround(q.x)), y = int(std::lround(q.y));
        if (x < 0 || y < 0 || x >= e.edges.cols || y >= e.edges.rows) continue;  // off the photo: no evidence
        ++inside;
        bool found = false;
        for (int j = -1; j <= 1 && !found; ++j)
            for (int i = -1; i <= 1 && !found; ++i) {
                const int xx = x + i, yy = y + j;
                if (xx < 0 || yy < 0 || xx >= e.edges.cols || yy >= e.edges.rows || !e.edges.at<uchar>(yy, xx))
                    continue;
                const double gx = e.dx.at<float>(yy, xx), gy = e.dy.at<float>(yy, xx), g = std::hypot(gx, gy);
                found = g > 0 && std::abs(gx * normal.x + gy * normal.y) >= cosTolerance * g;
            }
        supported += found;
    }
    return inside > 0 ? double(supported) / (steps + 1) : 0.0;
}

// Steps 1-3 for one choice of four sides, given as a cycle (each line meets the next one at a corner).
// nullopt when the shape is implausible or a side is not an edge.
std::optional<QuadCandidate> evaluate(const std::vector<HoughLine>& L, const std::array<size_t, 4>& cycle,
                                      const EdgeMap& e, const QuadParams& p) {
    Quad corners;
    if (!quadFromLines(L[cycle[0]], L[cycle[1]], L[cycle[2]], L[cycle[3]], corners)) return std::nullopt;
    if (!isPlausible(corners, e.edges.size(), p)) return std::nullopt;
    QuadCandidate q;
    q.lines = cycle;
    q.corners = orderCorners({corners.begin(), corners.end()});
    double product = 1;
    for (size_t i = 0; i < 4; ++i) {
        q.support[i] = sideSupport(q.corners[i], q.corners[(i + 1) % 4], e);
        if (q.support[i] < p.minSupport) return std::nullopt;
        product *= q.support[i];
    }
    q.area = quadArea(q.corners);
    q.score = q.area * product;
    return q;
}

// Step 5a: where the intensity steps across the side near `centre` (t along the side, n across).
// I is sampled every half pixel over ± 8 px along n, each sample averaged over 5 px along t to beat
// the noise; the step is where |dI/ds| peaks, refined by the parabola through its neighbours.
std::optional<cv::Point2d> stepPosition(const cv::Mat& grey, cv::Point2d centre, cv::Point2d t, cv::Point2d n) {
    constexpr int kHalfWidth = 16;  // 16 half-pixels = 8 px each way
    std::vector<double> I;
    for (int s = -kHalfWidth; s <= kHalfWidth; ++s) {
        double sum = 0;
        for (int along = -2; along <= 2; ++along) {
            const cv::Point2d q = centre + n * (0.5 * s) + t * double(along);
            sum += vl::bilinear(grey, float(q.x), float(q.y));
        }
        I.push_back(sum / 5);
    }
    auto slope = [&](int s) { return std::abs(I[size_t(s + 1)] - I[size_t(s - 1)]); };  // central difference
    int peak = 1;
    for (int s = 1; s + 1 < int(I.size()); ++s)
        if (slope(s) > slope(peak)) peak = s;
    if (peak < 2 || peak + 2 >= int(I.size())) return std::nullopt;  // no step inside the window
    const double cm = slope(peak - 1), c0 = slope(peak), cp = slope(peak + 1);
    const double curvature = cm - 2 * c0 + cp;
    const double offset = curvature < 0 ? 0.5 * (cm - cp) / curvature : 0.0;
    return centre + n * (0.5 * (peak - kHalfWidth + offset));
}

// Step 5b: total least squares. The best line passes through the centroid c along the main axis of
// the scatter, so the distance minimised is perpendicular to the line (not vertical, as in y = a x + b):
// with S = Σ (q − c)(q − c)ᵀ the axis angle is φ = ½ atan2(2 S_xy, S_xx − S_yy) and the normal φ + 90°.
HoughLine fitLine(const std::vector<cv::Point2d>& points) {
    cv::Point2d c(0, 0);
    for (const auto& q : points) c += q;
    c *= 1.0 / double(points.size());
    double sxx = 0, sxy = 0, syy = 0;
    for (const auto& q : points) {
        sxx += (q.x - c.x) * (q.x - c.x);
        sxy += (q.x - c.x) * (q.y - c.y);
        syy += (q.y - c.y) * (q.y - c.y);
    }
    const double theta = 0.5 * std::atan2(2 * sxy, sxx - syy) + std::numbers::pi / 2;
    return {theta, c.x * std::cos(theta) + c.y * std::sin(theta), int(points.size())};
}

double distance(const HoughLine& l, cv::Point2d q) {
    return std::abs(q.x * std::cos(l.theta) + q.y * std::sin(l.theta) - l.rho);
}

}  // namespace

double quadArea(const Quad& q) {
    double twice = 0;
    for (size_t i = 0; i < 4; ++i) twice += q[i].cross(q[(i + 1) % 4]);
    return std::abs(twice) / 2;
}

std::optional<QuadCandidate> bestQuadrilateral(const std::vector<HoughLine>& L, const EdgeMap& e, const QuadParams& p) {
    const size_t n = L.size();
    std::optional<QuadCandidate> best;
    for (size_t a = 0; a < n; ++a)
        for (size_t b = a + 1; b < n; ++b)
            for (size_t c = b + 1; c < n; ++c)
                for (size_t d = c + 1; d < n; ++d) {
                    // The three ways to split four lines into two pairs of opposite sides, as cycles:
                    // opposite (a, b) and (c, d) → a c b d;  (a, c), (b, d) → a b c d;  (a, d), (b, c) → a b d c.
                    const std::array<std::array<size_t, 4>, 3> cycles{{{a, c, b, d}, {a, b, c, d}, {a, b, d, c}}};
                    for (const auto& cycle : cycles) {
                        const auto q = evaluate(L, cycle, e, p);
                        if (q && (!best || q->score > best->score)) best = q;
                    }
                }
    return best;
}

HoughLine refineSide(const cv::Mat& grey, cv::Point2d a, cv::Point2d b) {
    const cv::Point2d d = b - a;
    const cv::Point2d t = d / std::hypot(d.x, d.y), n(-t.y, t.x);
    // 48 points along the middle 80 % of the side: corners are often blurred or covered.
    std::vector<cv::Point2d> points;
    for (int k = 0; k < 48; ++k)
        if (const auto q = stepPosition(grey, a + d * (0.1 + 0.8 * k / 47.0), t, n)) points.push_back(*q);
    // Fit twice: first the points within 4 px of the coarse side, then those within 1.5 px of that
    // fit, which drops an object lying over the edge.
    HoughLine line{std::atan2(n.y, n.x), a.x * n.x + a.y * n.y, 0};
    for (const double tolerance : {4.0, 1.5}) {
        std::vector<cv::Point2d> kept;
        for (const auto& q : points)
            if (distance(line, q) <= tolerance) kept.push_back(q);
        if (kept.size() < 8) break;  // too little evidence: keep what we have
        line = fitLine(kept);
    }
    return line;
}

Quad orderCorners(const std::vector<cv::Point2d>& points) {
    CV_Assert(points.size() == 4);
    cv::Point2d centre(0, 0);
    for (const auto& p : points) centre += p * 0.25;
    std::vector<cv::Point2d> byAngle = points;
    // atan2 grows clockwise on screen because y points down.
    std::sort(byAngle.begin(), byAngle.end(), [&](cv::Point2d u, cv::Point2d v) {
        return std::atan2(u.y - centre.y, u.x - centre.x) < std::atan2(v.y - centre.y, v.x - centre.x);
    });
    size_t first = 0;
    for (size_t i = 1; i < 4; ++i)
        if (byAngle[i].x + byAngle[i].y < byAngle[first].x + byAngle[first].y) first = i;
    Quad q;
    for (size_t i = 0; i < 4; ++i) q[i] = byAngle[(first + i) % 4];
    return q;
}

}  // namespace vl::scanner
