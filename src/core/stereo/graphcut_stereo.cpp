#include "core/stereo/graphcut_stereo.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <string>

#include <opencv2/imgproc.hpp>

#include "core/stereo/maxflow.hpp"

namespace vl::stereo {
namespace {

// Costs are fixed-point integers (1/16 of a grey level), so that the minimum cut is exact and the
// energy never rises through rounding; they are converted back to doubles only for reporting.
using Cost = std::int64_t;
constexpr double kScale = 16;
constexpr Cost kInf = Cost(1) << 50;  // forbids a cut; far above any energy

Cost toCost(double v) { return Cost(std::llround(v * kScale)); }

constexpr int kOcc = std::numeric_limits<int>::min();  // "no active assignment" for a pixel
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

}  // namespace

struct GraphCutStereo::Impl {
    // Float copy of an image, channels interleaved, plus the Birchfield–Tomasi intervals: for each pixel
    // the range of the linearly interpolated signal within half a pixel to the left and right.
    struct Channels {
        int W = 0, H = 0, C = 0;
        std::vector<float> v, lo, hi, grey;

        explicit Channels(const cv::Mat& img) {
            CV_Assert(img.depth() == CV_8U && (img.channels() == 1 || img.channels() == 3));
            W = img.cols;
            H = img.rows;
            C = img.channels();
            const std::size_t n = std::size_t(W) * std::size_t(H) * std::size_t(C);
            v.resize(n);
            lo.resize(n);
            hi.resize(n);
            grey.resize(std::size_t(W) * std::size_t(H));
            for (int y = 0; y < H; ++y) {
                const uchar* row = img.ptr<uchar>(y);
                for (int i = 0; i < W * C; ++i) v[std::size_t(y * W * C + i)] = row[i];
            }
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    for (int c = 0; c < C; ++c) {
                        const std::size_t k = std::size_t((y * W + x) * C + c);
                        const float m = v[k];
                        const float l = x > 0 ? 0.5f * (m + v[k - std::size_t(C)]) : m;
                        const float r = x + 1 < W ? 0.5f * (m + v[k + std::size_t(C)]) : m;
                        lo[k] = std::min({m, l, r});
                        hi[k] = std::max({m, l, r});
                    }
            cv::Mat g;
            if (C == 3)
                cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
            else
                g = img;
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) grey[std::size_t(y * W + x)] = g.at<uchar>(y, x);
        }
    };

    GraphCutParams p;
    Channels L, R;
    int W, H;
    std::vector<int> dL, dR;   // disparity of each left / right pixel's active assignment, or kOcc
    std::vector<int> nodeO, nodeA;  // per left pixel: node of its current assignment / its α-assignment
    Cost lambdaSame, lambdaEdge, K;  // V inside regions / across intensity edges, occlusion cost
    MaxFlow<Cost> graph;

    Impl(const cv::Mat& left, const cv::Mat& right, const GraphCutParams& params)
        : p(params), L(left), R(right), W(left.cols), H(left.rows), lambdaSame(toCost(p.lambda)),
          lambdaEdge(p.adaptiveLambda ? toCost(p.lambda / 3) : lambdaSame), K(toCost(p.occlusionCost)) {
        CV_Assert(left.size() == right.size() && left.type() == right.type());
        CV_Assert(p.minDisparity >= 0 && p.maxDisparity >= p.minDisparity);
        const std::size_t n = std::size_t(W) * std::size_t(H);
        dL.assign(n, p.modelOcclusions ? kOcc : p.minDisparity);
        dR.assign(n, kOcc);
        if (!p.modelOcclusions) dR.clear();
        nodeO.assign(n, -1);
        nodeA.assign(n, -1);
        graph.reserve(int(2 * n), int(6 * n));
    }

    int index(int x, int y) const { return y * W + x; }
    bool inRange(int x, int d) const { return x - d >= 0 && x - d < W; }

    // D(<(x, y), (x - d, y)>): mean over channels of the dissimilarity, truncated.
    Cost data(int x, int y, int d) const {
        if (!inRange(x, d)) return toCost(p.truncation);  // only reachable in Potts mode
        const int C = L.C;
        const std::size_t kl = std::size_t((y * W + x) * C), kr = std::size_t((y * W + x - d) * C);
        float sum = 0;
        for (int c = 0; c < C; ++c) {
            const float l = L.v[kl + std::size_t(c)], r = R.v[kr + std::size_t(c)];
            if (p.birchfieldTomasi) {
                // Distance from each value to the other image's interval, the smaller of the two.
                const float dlr = std::max({0.f, l - R.hi[kr + std::size_t(c)], R.lo[kr + std::size_t(c)] - l});
                const float drl = std::max({0.f, r - L.hi[kl + std::size_t(c)], L.lo[kl + std::size_t(c)] - r});
                sum += std::min(dlr, drl);
            } else {
                sum += std::abs(l - r);
            }
        }
        const double t = std::min(double(sum) / C, p.truncation);
        return toCost(p.cost == DataCost::Absolute ? t : t * t / p.truncation);
    }

    // V for the assignments of left pixels i0, i1 (neighbours) at disparity d.
    Cost smooth(int i0, int i1, int d) const {
        if (!p.adaptiveLambda) return lambdaSame;
        float diff = std::abs(L.grey[std::size_t(i0)] - L.grey[std::size_t(i1)]);
        if (p.modelOcclusions) diff = std::max(diff, std::abs(R.grey[std::size_t(i0 - d)] - R.grey[std::size_t(i1 - d)]));
        return diff < p.edgeThreshold ? lambdaSame : lambdaEdge;
    }

    // Calls f(i0, x0, i1, x1) for every pair of 4-neighbours (right and down).
    template <typename F>
    void forNeighbours(F&& f) const {
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const int i = index(x, y);
                if (x + 1 < W) f(i, x, i + 1, x + 1);
                if (y + 1 < H) f(i, x, i + W, x);
            }
    }

    void energyTerms(Cost& eData, Cost& eOcc, Cost& eSmooth) const {
        eData = eOcc = eSmooth = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const int d = dL[std::size_t(index(x, y))];
                if (d == kOcc)
                    eOcc += K;
                else
                    eData += data(x, y, d);
            }
        if (p.modelOcclusions) {
            for (int d : dR)
                if (d == kOcc) eOcc += K;
            // Assignment pair (p0, p0 - d), (p1, p1 - d) pays V when exactly one is active.
            forNeighbours([&](int i0, int x0, int i1, int x1) {
                const int d0 = dL[std::size_t(i0)], d1 = dL[std::size_t(i1)];
                if (d0 == d1) return;
                if (d0 != kOcc && inRange(x1, d0)) eSmooth += smooth(i0, i1, d0);
                if (d1 != kOcc && inRange(x0, d1)) eSmooth += smooth(i0, i1, d1);
            });
        } else {
            forNeighbours([&](int i0, int, int i1, int) {
                if (dL[std::size_t(i0)] != dL[std::size_t(i1)]) eSmooth += smooth(i0, i1, 0);
            });
        }
    }

    Cost energy() const {
        Cost a, b, c;
        energyTerms(a, b, c);
        return a + b + c;
    }

    // ---- binary energy on graph nodes: label 0 = source side, 1 = sink side
    void unary(int v, Cost e0, Cost e1) { graph.addTweights(v, e1, e0); }
    // A submodular pair E(x, y) with E(0,0) = A, E(0,1) = B, E(1,0) = C, E(1,1) = D, written as
    // A + (C-A)·x + (D-C)·y + (B+C-A-D)·[x = 0, y = 1] (Kolmogorov–Zabih 2002, "What energy functions
    // can be minimized via graph cuts?"). The constant A does not change the cut and is dropped.
    void pairwise(int x, int y, Cost A, Cost B, Cost C, Cost D) {
        unary(x, 0, C - A);
        unary(y, 0, D - C);
        graph.addEdge(x, y, B + C - A - D, 0);
    }

    // Occlusion and uniqueness at one pixel whose candidate assignments in the move are o (a current
    // one, label 1 = dropped) and a (the α one, label 1 = switched on); -1 when absent.
    void pixelTerm(int o, int a) {
        if (o >= 0 && a >= 0)
            graph.addEdge(o, a, kInf, K);  // keep o and switch a on: forbidden; drop o, a off: occluded
        else if (o >= 0)
            unary(o, 0, K);
        else if (a >= 0)
            unary(a, K, 0);
        // neither: the pixel stays occluded whatever the move does, a constant
    }

    // ---- one Kolmogorov–Zabih expansion move (section 5 of the paper)
    void buildAssignmentMove(int alpha) {
        // Nodes and data terms. Current assignments with disparity α are not in A0: their pixel's α
        // variable covers them (the paper's Aα contains every α-assignment, active or not).
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const std::size_t i = std::size_t(index(x, y));
                const int d = dL[i];
                nodeO[i] = nodeA[i] = -1;
                if (d != kOcc && d != alpha) {
                    nodeO[i] = graph.addNode();
                    unary(nodeO[i], data(x, y, d), 0);
                }
                if (inRange(x, alpha)) {
                    nodeA[i] = graph.addNode();
                    unary(nodeA[i], 0, data(x, y, alpha));
                }
            }
        // Occlusion + uniqueness, left pixels then right pixels. A right pixel xr meets at most one
        // current assignment (its holder) and one α-assignment (from the left pixel xr + α).
        for (std::size_t i = 0; i < dL.size(); ++i) pixelTerm(nodeO[i], nodeA[i]);
        for (int y = 0; y < H; ++y)
            for (int xr = 0; xr < W; ++xr) {
                const int d = dR[std::size_t(index(xr, y))];
                const int o = (d != kOcc && d != alpha) ? nodeO[std::size_t(index(xr + d, y))] : -1;
                const int a = xr + alpha < W ? nodeA[std::size_t(index(xr + alpha, y))] : -1;
                pixelTerm(o, a);
            }
        // Smoothness between assignments of equal disparity on neighbouring pixels.
        forNeighbours([&](int i0, int x0, int i1, int x1) {
            const std::size_t j0 = std::size_t(i0), j1 = std::size_t(i1);
            if (nodeA[j0] >= 0 && nodeA[j1] >= 0) {  // α layer: V·[on ≠ on]
                const Cost V = smooth(i0, i1, alpha);
                graph.addEdge(nodeA[j0], nodeA[j1], V, V);
            }
            const int d0 = dL[j0], d1 = dL[j1];
            if (nodeO[j0] >= 0 && inRange(x1, d0)) {
                const Cost V = smooth(i0, i1, d0);
                if (d1 == d0)
                    graph.addEdge(nodeO[j0], nodeO[j1], V, V);  // both current: V·[kept ≠ kept]
                else
                    unary(nodeO[j0], V, 0);  // partner inactive and stays so: V while kept
            }
            if (nodeO[j1] >= 0 && d1 != d0 && inRange(x0, d1)) unary(nodeO[j1], smooth(i0, i1, d1), 0);
        });
    }

    void applyAssignmentMove(int alpha) {
        for (std::size_t i = 0; i < dL.size(); ++i) {
            if (nodeO[i] >= 0 && graph.segment(nodeO[i]) == MaxFlow<Cost>::Sink) dL[i] = kOcc;
            if (nodeA[i] >= 0) {
                if (graph.segment(nodeA[i]) == MaxFlow<Cost>::Sink)
                    dL[i] = alpha;
                else if (dL[i] == alpha)
                    dL[i] = kOcc;
            }
        }
        rebuildRight();
    }

    void rebuildRight() {
        std::fill(dR.begin(), dR.end(), kOcc);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const int d = dL[std::size_t(index(x, y))];
                if (d != kOcc) dR[std::size_t(index(x - d, y))] = d;
            }
    }

    // ---- one Potts α-expansion on left labels (Boykov, Veksler, Zabih), label 1 = take α
    void buildPottsMove(int alpha) {
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const std::size_t i = std::size_t(index(x, y));
                nodeA[i] = -1;
                if (dL[i] == alpha) continue;
                nodeA[i] = graph.addNode();
                unary(nodeA[i], data(x, y, dL[i]), data(x, y, alpha));
            }
        forNeighbours([&](int i0, int, int i1, int) {
            const std::size_t j0 = std::size_t(i0), j1 = std::size_t(i1);
            const Cost V = smooth(i0, i1, 0);
            const int v0 = nodeA[j0], v1 = nodeA[j1];
            if (v0 >= 0 && v1 >= 0)
                pairwise(v0, v1, dL[j0] != dL[j1] ? V : 0, V, V, 0);
            else if (v0 >= 0)
                unary(v0, V, 0);  // neighbour already at α
            else if (v1 >= 0)
                unary(v1, V, 0);
        });
    }

    void applyPottsMove(int alpha) {
        for (std::size_t i = 0; i < dL.size(); ++i)
            if (nodeA[i] >= 0 && graph.segment(nodeA[i]) == MaxFlow<Cost>::Sink) dL[i] = alpha;
    }

    bool expand(int alpha) {
        const Cost before = energy();
        const std::vector<int> saved = dL;
        graph.clear();
        if (p.modelOcclusions)
            buildAssignmentMove(alpha);
        else
            buildPottsMove(alpha);
        graph.maxflow();
        if (p.modelOcclusions)
            applyAssignmentMove(alpha);
        else
            applyPottsMove(alpha);
        // The cut is the exact optimum of the move and the current configuration is one of the
        // candidates, so the energy cannot rise; on a tie keep the old map.
        if (energy() < before) return true;
        dL = saved;
        if (p.modelOcclusions) rebuildRight();
        return false;
    }

    cv::Mat disparity() const {
        cv::Mat out(H, W, CV_32F);
        for (int y = 0; y < H; ++y) {
            float* row = out.ptr<float>(y);
            for (int x = 0; x < W; ++x) {
                const int d = dL[std::size_t(index(x, y))];
                row[x] = d == kOcc ? kNaN : float(d);
            }
        }
        return out;
    }
};

GraphCutStereo::GraphCutStereo(const cv::Mat& left, const cv::Mat& right, const GraphCutParams& p)
    : impl_(std::make_unique<Impl>(left, right, p)) {}
GraphCutStereo::~GraphCutStereo() = default;

bool GraphCutStereo::expand(int alpha) { return impl_->expand(alpha); }
double GraphCutStereo::energy() const { return double(impl_->energy()) / kScale; }
void GraphCutStereo::energyTerms(double& data, double& occlusion, double& smooth) const {
    Cost a, b, c;
    impl_->energyTerms(a, b, c);
    data = double(a) / kScale;
    occlusion = double(b) / kScale;
    smooth = double(c) / kScale;
}
cv::Mat GraphCutStereo::disparity() const { return impl_->disparity(); }

cv::Mat GraphCutStereo::occludedRight() const {
    cv::Mat out = cv::Mat::zeros(impl_->H, impl_->W, CV_8U);
    if (!impl_->p.modelOcclusions) return out;
    for (int y = 0; y < out.rows; ++y)
        for (int x = 0; x < out.cols; ++x)
            if (impl_->dR[std::size_t(y * out.cols + x)] == kOcc) out.at<uchar>(y, x) = 255;
    return out;
}

bool GraphCutStereo::setDisparity(const cv::Mat& disparity) {
    Impl& m = *impl_;
    CV_Assert(disparity.type() == CV_32F && disparity.rows == m.H && disparity.cols == m.W);
    std::vector<int> d(m.dL.size());
    std::vector<char> taken(m.dL.size(), 0);
    for (int y = 0; y < m.H; ++y)
        for (int x = 0; x < m.W; ++x) {
            const float v = disparity.at<float>(y, x);
            int& di = d[std::size_t(m.index(x, y))];
            if (!std::isfinite(v)) {
                if (!m.p.modelOcclusions) return false;
                di = kOcc;
                continue;
            }
            di = int(std::lround(v));
            if (di < m.p.minDisparity || di > m.p.maxDisparity) return false;
            if (!m.p.modelOcclusions) continue;
            if (!m.inRange(x, di)) return false;
            char& t = taken[std::size_t(m.index(x - di, y))];
            if (t) return false;  // two left pixels on one right pixel
            t = 1;
        }
    m.dL = std::move(d);
    if (m.p.modelOcclusions) m.rebuildRight();
    return true;
}

GraphCutResult graphCutStereo(const cv::Mat& left, const cv::Mat& right, const GraphCutParams& p,
                              std::stop_token stop, const ProgressFn& progress, const GraphCutObserver& observer) {
    GraphCutStereo solver(left, right, p);
    std::vector<int> labels(std::size_t(p.maxDisparity - p.minDisparity + 1));
    std::iota(labels.begin(), labels.end(), p.minDisparity);
    std::mt19937 rng(p.seed);

    GraphCutResult res;
    res.energy.push_back(solver.energy());
    const int total = std::max(1, p.maxIterations) * int(labels.size());
    for (int it = 1; it <= p.maxIterations; ++it) {
        if (p.randomOrder) std::shuffle(labels.begin(), labels.end(), rng);
        bool any = false;
        for (int alpha : labels) {
            checkStop(stop);
            const bool improved = solver.expand(alpha);
            any |= improved;
            ++res.moves;
            res.improvingMoves += improved ? 1 : 0;
            res.energy.push_back(solver.energy());
            report(progress, double(res.moves) / total,
                   "pass " + std::to_string(it) + ", expanding disparity " + std::to_string(alpha));
            if (observer) {
                const cv::Mat d = solver.disparity();
                observer({it, alpha, res.moves, improved, res.energy.back(), &d});
            }
        }
        res.iterations = it;
        if (!any) {
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
    solver.energyTerms(res.dataEnergy, res.occlusionEnergy, res.smoothEnergy);
    report(progress, 1.0, "done");
    return res;
}

double graphCutEnergy(const cv::Mat& left, const cv::Mat& right, const cv::Mat& disparity, const GraphCutParams& p) {
    GraphCutStereo s(left, right, p);
    if (!s.setDisparity(disparity)) return std::numeric_limits<double>::infinity();
    return s.energy();
}

cv::Mat occlusionsFromDisparity(const cv::Mat& truth) {
    CV_Assert(truth.type() == CV_32F);
    cv::Mat occ = cv::Mat::zeros(truth.size(), CV_8U);
    for (int y = 0; y < truth.rows; ++y) {
        const float* t = truth.ptr<float>(y);
        uchar* o = occ.ptr<uchar>(y);
        // Sweep right to left keeping the leftmost right-image column reached by pixels further right.
        float reach = std::numeric_limits<float>::infinity();
        for (int x = truth.cols - 1; x >= 0; --x) {
            if (t[x] <= 0) continue;
            const float xr = float(x) - t[x];
            if (xr < -0.5f || reach < xr - 0.5f) o[x] = 255;
            reach = std::min(reach, xr);
        }
    }
    return occ;
}

}  // namespace vl::stereo
