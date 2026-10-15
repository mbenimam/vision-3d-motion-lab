// Stereo with occlusions by α-expansion and graph cuts (Kolmogorov and Zabih, ICCV 2001).
// Course: "3D Vision and Motion", part 8 (the energy and the moves) on top of part 7 (max-flow).
//
// The algorithm (graphCutStereo):
//   start from the configuration with every pixel occluded; repeat passes until one changes nothing:
//     for each disparity α (in random order): expand(α)
//
// One α-expansion (expand), as a graph with one binary variable per assignment that may change:
//   1. addAssignmentNodes  the variables, and E_data on each
//   2. addOcclusionTerms   E_occ and uniqueness, one term per pixel of either image
//   3. addSmoothnessTerms  E_smooth, one term per pair of neighbouring assignments
//   4. MaxFlow::solve      the minimum cut is the move of least energy
//   5. applyAssignmentMove read the new configuration off the cut
// Every term is written as the table of its values on the variables it involves; addUnary and
// addPairwise turn a table into terminal edges and one arc (Kolmogorov and Zabih 2004), which is
// possible exactly when the table is regular: E(0,0) + E(1,1) <= E(0,1) + E(1,0).
//
// Costs are doubles. The cut is the exact optimum up to rounding, and a move is accepted only if it
// lowers E by more than rounding error, so the energy never increases.

#include "core/stereo/graphcut_stereo.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <string>

namespace vl::stereo {
namespace {

constexpr int kNoNode = -1;
constexpr double kInfinity = std::numeric_limits<double>::infinity();
const float kNaN = std::numeric_limits<float>::quiet_NaN();

// ---- binary energies as graphs (Kolmogorov and Zabih 2004, "What energy functions can be
// minimized via graph cuts?"). Each node v is a binary variable: x_v = 0 when the cut leaves v on
// the source side, x_v = 1 on the sink side.

// E_v(x_v) with E_v(0) = E0, E_v(1) = E1: the cut severs v → t when x_v = 0 and s → v when x_v = 1.
void addUnary(MaxFlow& graph, int v, double E0, double E1) { graph.addTerminalEdges(v, E1, E0); }

// E(x_v, x_w) given by its table. Rewritten as
//   E00 + (E10 − E00)·x_v + (E11 − E10)·x_w + (E01 + E10 − E00 − E11)·(1 − x_v)·x_w
// it is a constant (dropped: the same for every cut), two unary terms, and the arc v → w, which a
// cut severs exactly when x_v = 0 and x_w = 1. An arc needs a capacity >= 0: that is regularity.
void addPairwise(MaxFlow& graph, int v, int w, double E00, double E01, double E10, double E11) {
    CV_DbgAssert(E00 + E11 <= E01 + E10);
    addUnary(graph, v, 0, E10 - E00);
    addUnary(graph, w, 0, E11 - E10);
    graph.addEdge(v, w, E01 + E10 - E00 - E11, 0);
}

}  // namespace

GraphCutStereo::GraphCutStereo(const cv::Mat& left, const cv::Mat& right, const GraphCutParams& p)
    : E_(left, right, p) {
    const int W = left.cols, H = left.rows;
    if (p.modelOcclusions) {
        f_.left = cv::Mat(H, W, CV_32S, cv::Scalar(kOccluded));
        f_.right = cv::Mat(H, W, CV_32S, cv::Scalar(kOccluded));
    } else {
        f_.left = cv::Mat(H, W, CV_32S, cv::Scalar(p.minDisparity));
    }
    nodeCurrent_ = cv::Mat(H, W, CV_32S, cv::Scalar(kNoNode));
    nodeAlpha_ = cv::Mat(H, W, CV_32S, cv::Scalar(kNoNode));
    graph_.reserve(2 * W * H, 6 * W * H);
}

bool GraphCutStereo::expand(int alpha) {
    const double before = energy();
    const Configuration saved{f_.left.clone(), f_.right.clone()};
    graph_.clear();
    if (E_.params().modelOcclusions) {
        addAssignmentNodes(alpha);   // 1.
        addOcclusionTerms(alpha);    // 2.
        addSmoothnessTerms(alpha);   // 3.
        graph_.solve();              // 4.
        applyAssignmentMove(alpha);  // 5.
    } else {
        buildPottsMove(alpha);
        graph_.solve();
        applyPottsMove(alpha);
    }
    // The current configuration is one of the candidates, so the cut cannot be worse except by
    // rounding; keep the old configuration unless the new one is really better.
    if (energy() < before - 1e-9 * std::max(1.0, before)) return true;
    f_ = saved;
    return false;
}

// ---- the Kolmogorov–Zabih expansion move

// Step 1. One variable per assignment the move may switch (the paper's section 5.2):
//   A0 = active assignments with d ≠ α   x = 0: stays active,  x = 1: dropped
//   Aα = all assignments with d = α      x = 1: active,        x = 0: off
// (The opposite conventions are what make the terms of step 2 regular.) E_data on each, D(a) paid
// when a is active:  a ∈ A0: E(0) = D(a), E(1) = 0;   a ∈ Aα: E(0) = 0, E(1) = D(a).
void GraphCutStereo::addAssignmentNodes(int alpha) {
    for (int y = 0; y < E_.height(); ++y)
        for (int x = 0; x < E_.width(); ++x) {
            int& current = nodeCurrent_.at<int>(y, x);
            int& candidate = nodeAlpha_.at<int>(y, x);
            current = candidate = kNoNode;
            const int d = f_.left.at<int>(y, x);
            if (d != kOccluded && d != alpha) {
                current = graph_.addNode();
                addUnary(graph_, current, E_.D(x, y, d), 0);
            }
            if (E_.exists(x, alpha)) {
                candidate = graph_.addNode();
                addUnary(graph_, candidate, 0, E_.D(x, y, alpha));
            }
        }
}

// Step 2, for one pixel. A pixel meets at most two assignments of the move: its current one c (in A0)
// and its α one a (in Aα).
//   x_c x_a | the pixel is      | E
//    0   0  | matched by c      | 0
//    0   1  | matched twice     | +∞  (not unique)
//    1   0  | unmatched         | K
//    1   1  | matched by a      | 0
// Regular: 0 + 0 <= ∞ + K. With only one of them, the pixel is unmatched when that one is off.
// With neither, the pixel's state cannot change: a constant, left out.
void GraphCutStereo::addOcclusionTerm(int c, int a) {
    const double K = E_.K();
    if (c != kNoNode && a != kNoNode)
        addPairwise(graph_, c, a, 0, kInfinity, K, 0);
    else if (c != kNoNode)
        addUnary(graph_, c, 0, K);
    else if (a != kNoNode)
        addUnary(graph_, a, K, 0);
}

void GraphCutStereo::addOcclusionTerms(int alpha) {
    const int W = E_.width();
    for (int y = 0; y < E_.height(); ++y)
        for (int x = 0; x < W; ++x) {
            // Left pixel (x, y): its own two assignments.
            addOcclusionTerm(nodeCurrent_.at<int>(y, x), nodeAlpha_.at<int>(y, x));
            // Right pixel (x, y): its current assignment comes from left pixel x + d, its α one from x + α.
            const int d = f_.right.at<int>(y, x);
            const int c = d != kOccluded && d != alpha ? nodeCurrent_.at<int>(y, x + d) : kNoNode;
            const int a = x + alpha < W ? nodeAlpha_.at<int>(y, x + alpha) : kNoNode;
            addOcclusionTerm(c, a);
        }
}

// Step 3, for the neighbouring left pixels p0 = (x0, y0), p1 = (x1, y1). E_smooth pays V when exactly
// one assignment of a pair {<p0, p0 − d>, <p1, p1 − d>} is active. The pairs the move can change:
//   d = α, both in Aα:          V·[x0 ≠ x1]   table 0, V, V, 0
//   d = d0 = d1 ≠ α, both in A0: V·[x0 ≠ x1]   table 0, V, V, 0
//   d = d0 ≠ d1: <p0, ·> in A0, its partner is off and not in the move (d0 ≠ α), so it stays off:
//                V while <p0, ·> stays active, V·[x0 = 0]; and the same for d = d1.
// Both tables are regular (0 + 0 <= V + V). A0 and Aα never pair: a pair has one disparity.
void GraphCutStereo::addSmoothnessTerm(int x0, int y0, int x1, int y1, int alpha) {
    const int a0 = nodeAlpha_.at<int>(y0, x0), a1 = nodeAlpha_.at<int>(y1, x1);
    if (a0 != kNoNode && a1 != kNoNode) {
        const double V = E_.V(x0, y0, x1, y1, alpha);
        addPairwise(graph_, a0, a1, 0, V, V, 0);
    }
    const int d0 = f_.left.at<int>(y0, x0), d1 = f_.left.at<int>(y1, x1);
    const int c0 = nodeCurrent_.at<int>(y0, x0), c1 = nodeCurrent_.at<int>(y1, x1);
    if (c0 != kNoNode && E_.exists(x1, d0)) {
        const double V = E_.V(x0, y0, x1, y1, d0);
        if (d1 == d0)
            addPairwise(graph_, c0, c1, 0, V, V, 0);
        else
            addUnary(graph_, c0, V, 0);
    }
    if (c1 != kNoNode && d1 != d0 && E_.exists(x0, d1)) addUnary(graph_, c1, E_.V(x0, y0, x1, y1, d1), 0);
}

void GraphCutStereo::addSmoothnessTerms(int alpha) {
    for (int y = 0; y < E_.height(); ++y)
        for (int x = 0; x < E_.width(); ++x) {
            if (x + 1 < E_.width()) addSmoothnessTerm(x, y, x + 1, y, alpha);
            if (y + 1 < E_.height()) addSmoothnessTerm(x, y, x, y + 1, alpha);
        }
}

// Step 5. An A0 assignment survives on the source side; an α-assignment is active on the sink side.
// The infinite edges of step 2 make the result unique.
void GraphCutStereo::applyAssignmentMove(int alpha) {
    for (int y = 0; y < E_.height(); ++y)
        for (int x = 0; x < E_.width(); ++x) {
            const int c = nodeCurrent_.at<int>(y, x), a = nodeAlpha_.at<int>(y, x);
            int& d = f_.left.at<int>(y, x);
            if (c != kNoNode && onSinkSide(c)) d = kOccluded;
            if (a != kNoNode && onSinkSide(a)) d = alpha;
            if (a != kNoNode && !onSinkSide(a) && d == alpha) d = kOccluded;
        }
    rebuildRight();
}

// The right image's view of the same assignments.
void GraphCutStereo::rebuildRight() {
    f_.right.setTo(kOccluded);
    for (int y = 0; y < E_.height(); ++y)
        for (int x = 0; x < E_.width(); ++x) {
            const int d = f_.left.at<int>(y, x);
            if (d != kOccluded) f_.right.at<int>(y, x - d) = d;
        }
}

// ---- the Potts expansion move (Boykov, Veksler, Zabih 2001)

// One variable per left pixel p not already at α; x_p = 1: p takes α.
//   unary:      E(0) = D(p, d_p), E(1) = D(p, α)
//   neighbours: E00 = V·[d_p ≠ d_q], E01 = E10 = V (d_p, d_q ≠ α), E11 = 0; regular as V·[d_p ≠ d_q] <= 2V
//   neighbour q already at α: V·[x_p = 0]
void GraphCutStereo::buildPottsMove(int alpha) {
    for (int y = 0; y < E_.height(); ++y)
        for (int x = 0; x < E_.width(); ++x) {
            const int d = f_.left.at<int>(y, x);
            int& v = nodeAlpha_.at<int>(y, x);
            v = kNoNode;
            if (d == alpha) continue;
            v = graph_.addNode();
            addUnary(graph_, v, E_.D(x, y, d), E_.D(x, y, alpha));
        }
    auto pair = [&](int x0, int y0, int x1, int y1) {
        const int v0 = nodeAlpha_.at<int>(y0, x0), v1 = nodeAlpha_.at<int>(y1, x1);
        const double V = E_.V(x0, y0, x1, y1, 0);
        const double differ = f_.left.at<int>(y0, x0) != f_.left.at<int>(y1, x1) ? V : 0;
        if (v0 != kNoNode && v1 != kNoNode) addPairwise(graph_, v0, v1, differ, V, V, 0);
        if (v0 != kNoNode && v1 == kNoNode) addUnary(graph_, v0, V, 0);
        if (v0 == kNoNode && v1 != kNoNode) addUnary(graph_, v1, V, 0);
    };
    for (int y = 0; y < E_.height(); ++y)
        for (int x = 0; x < E_.width(); ++x) {
            if (x + 1 < E_.width()) pair(x, y, x + 1, y);
            if (y + 1 < E_.height()) pair(x, y, x, y + 1);
        }
}

void GraphCutStereo::applyPottsMove(int alpha) {
    for (int y = 0; y < E_.height(); ++y)
        for (int x = 0; x < E_.width(); ++x) {
            const int v = nodeAlpha_.at<int>(y, x);
            if (v != kNoNode && onSinkSide(v)) f_.left.at<int>(y, x) = alpha;
        }
}

// ---- configuration in and out

cv::Mat GraphCutStereo::disparity() const {
    cv::Mat out(f_.left.size(), CV_32F);
    for (int y = 0; y < out.rows; ++y)
        for (int x = 0; x < out.cols; ++x) {
            const int d = f_.left.at<int>(y, x);
            out.at<float>(y, x) = d == kOccluded ? kNaN : float(d);
        }
    return out;
}

cv::Mat GraphCutStereo::occludedRight() const {
    cv::Mat out = cv::Mat::zeros(f_.left.size(), CV_8U);
    if (f_.right.empty()) return out;
    for (int y = 0; y < out.rows; ++y)
        for (int x = 0; x < out.cols; ++x)
            if (f_.right.at<int>(y, x) == kOccluded) out.at<uchar>(y, x) = 255;
    return out;
}

bool GraphCutStereo::setDisparity(const cv::Mat& disparity) {
    const GraphCutParams& p = E_.params();
    CV_Assert(disparity.type() == CV_32F && disparity.size() == f_.left.size());
    cv::Mat left(disparity.size(), CV_32S);
    cv::Mat taken = cv::Mat::zeros(disparity.size(), CV_8U);  // right pixels already matched
    for (int y = 0; y < left.rows; ++y)
        for (int x = 0; x < left.cols; ++x) {
            const float v = disparity.at<float>(y, x);
            if (!std::isfinite(v)) {
                if (!p.modelOcclusions) return false;
                left.at<int>(y, x) = kOccluded;
                continue;
            }
            const int d = int(std::lround(v));
            if (d < p.minDisparity || d > p.maxDisparity) return false;
            left.at<int>(y, x) = d;
            if (!p.modelOcclusions) continue;
            if (!E_.exists(x, d) || taken.at<uchar>(y, x - d)) return false;  // no partner, or not unique
            taken.at<uchar>(y, x - d) = 1;
        }
    f_.left = left;
    if (p.modelOcclusions) rebuildRight();
    return true;
}

// ---- the outer loop

GraphCutResult graphCutStereo(const cv::Mat& left, const cv::Mat& right, const GraphCutParams& p,
                              std::stop_token stop, const ProgressFn& progress, const GraphCutObserver& observer) {
    GraphCutStereo solver(left, right, p);
    std::vector<int> alphas(std::size_t(p.maxDisparity - p.minDisparity + 1));
    std::iota(alphas.begin(), alphas.end(), p.minDisparity);
    std::mt19937 rng(p.seed);

    GraphCutResult res;
    res.energy.push_back(solver.energy());
    const int total = std::max(1, p.maxIterations) * int(alphas.size());
    for (int pass = 1; pass <= p.maxIterations; ++pass) {
        // A random order avoids favouring the disparities expanded first (the paper's advice).
        if (p.randomOrder) std::shuffle(alphas.begin(), alphas.end(), rng);
        bool anyImproved = false;
        for (int alpha : alphas) {
            checkStop(stop);
            const bool improved = solver.expand(alpha);
            anyImproved = anyImproved || improved;
            ++res.moves;
            res.improvingMoves += improved ? 1 : 0;
            res.energy.push_back(solver.energy());
            report(progress, double(res.moves) / total,
                   "pass " + std::to_string(pass) + ", expanding disparity " + std::to_string(alpha));
            if (observer) {
                const cv::Mat d = solver.disparity();
                observer({pass, alpha, res.moves, improved, res.energy.back(), &d});
            }
        }
        res.iterations = pass;
        if (!anyImproved) {
            res.converged = true;
            break;
        }
    }
    res.disparity = solver.disparity();
    res.occluded = cv::Mat::zeros(res.disparity.size(), CV_8U);
    for (int y = 0; y < res.disparity.rows; ++y)
        for (int x = 0; x < res.disparity.cols; ++x)
            if (std::isnan(res.disparity.at<float>(y, x))) res.occluded.at<uchar>(y, x) = 255;
    res.occludedRight = solver.occludedRight();
    res.terms = solver.energyTerms();
    report(progress, 1.0, "done");
    return res;
}

double graphCutEnergy(const cv::Mat& left, const cv::Mat& right, const cv::Mat& disparity, const GraphCutParams& p) {
    GraphCutStereo s(left, right, p);
    if (!s.setDisparity(disparity)) return std::numeric_limits<double>::infinity();
    return s.energy();
}

// Sweep each row from right to left, keeping the leftmost right-image column reached so far by the
// pixels further right (which are in front when they land further left). A pixel whose match lies
// right of that column is hidden behind them; one whose match falls off the image is hidden too.
cv::Mat occlusionsFromDisparity(const cv::Mat& truth) {
    CV_Assert(truth.type() == CV_32F);
    cv::Mat occ = cv::Mat::zeros(truth.size(), CV_8U);
    for (int y = 0; y < truth.rows; ++y) {
        float reach = std::numeric_limits<float>::infinity();
        for (int x = truth.cols - 1; x >= 0; --x) {
            const float d = truth.at<float>(y, x);
            if (d <= 0) continue;  // unknown
            const float xr = float(x) - d;
            if (xr < -0.5f || reach < xr - 0.5f) occ.at<uchar>(y, x) = 255;
            reach = std::min(reach, xr);
        }
    }
    return occ;
}

}  // namespace vl::stereo
