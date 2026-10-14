// Horn–Schunck optical flow, coarse to fine with warping. Course: "3D Vision and Motion", part 11.
//
// The method, one function per step:
//   1. buildPyramid        both frames, halved levels − 1 times (flow_common)
//   2. upsampleFlow        coarsest level first; each level starts from the coarser answer, doubled
//   3. linearise           warp I1 by the current flow w0 = (u0, v0), take Ix, Iy, It (flow_common),
//                          and fold w0 into the data offset It0 = It − Ix·u0 − Iy·v0
//   4. hornSchunckSweep    one Jacobi or Gauss–Seidel pass of the Euler–Lagrange update over all pixels
//   5. hornSchunckEnergy   E after each sweep at the finest level, to watch it converge
// Steps 3 and 4 alternate: `warps` linearisations per level, the sweeps split between them (solveLevel).
//
// Discretisation of E. The smoothness term is a sum over pairs of neighbours with Horn and Schunck's
// weights, ω = 1/6 for the 4 direct neighbours and 1/12 for the 4 diagonals:
//     E = Σ_x (Ix·u + Iy·v + It0)²  +  α² Σ_{pairs x, x'} ω·((u(x) − u(x'))² + (v(x) − v(x'))²)
//
// Euler–Lagrange equations. ∂E/∂u(x) = ∂E/∂v(x) = 0 at one pixel gives, with ū, v̄ the ω-weighted
// mean of its neighbours and s = Σ ω (1 inside the image, less on the border):
//     Ix·(Ix·u + Iy·v + It0) + α²·s·(u − ū) = 0
//     Iy·(Ix·u + Iy·v + It0) + α²·s·(v − v̄) = 0
// (the discrete form of Ix·(Ix·u + Iy·v + It) = α²·Δu, with Δu ≈ s·(ū − u)). Solved for (u, v) with
// the neighbours fixed:
//     t = (Ix·ū + Iy·v̄ + It0) / (α²·s + Ix² + Iy²),   u = ū − Ix·t,   v = v̄ − Iy·t
// Jacobi computes ū, v̄ from the previous sweep for every pixel; Gauss–Seidel uses the new values
// as soon as they exist. Gauss–Seidel is exact coordinate descent (E never increases) and converges
// about twice as fast.
//
// Cost: O(levels · iterations · W·H).

#include "core/flow/horn_schunck.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace vl::flow {

namespace {

constexpr float kDirect = 1.f / 6.f, kDiagonal = 1.f / 12.f;

struct Neighbour {
    int dx, dy;
    float weight;
};
constexpr Neighbour kNeighbours[8] = {{-1, 0, kDirect},    {1, 0, kDirect},     {0, -1, kDirect},
                                      {0, 1, kDirect},     {-1, -1, kDiagonal}, {1, -1, kDiagonal},
                                      {-1, 1, kDiagonal},  {1, 1, kDiagonal}};

// ū, v̄ and s at (x, y): the weighted mean over the neighbours that exist, and their total weight.
struct NeighbourMean {
    float u = 0, v = 0, s = 0;
};

NeighbourMean neighbourMean(const cv::Mat& u, const cv::Mat& v, int y, int x) {
    NeighbourMean m;
    for (const Neighbour& n : kNeighbours) {
        const int xx = x + n.dx, yy = y + n.dy;
        if (xx < 0 || yy < 0 || xx >= u.cols || yy >= u.rows) continue;
        m.u += n.weight * u.at<float>(yy, xx);
        m.v += n.weight * v.at<float>(yy, xx);
        m.s += n.weight;
    }
    m.u /= m.s;
    m.v /= m.s;
    return m;
}

}  // namespace

double hornSchunckSweep(cv::Mat& u, cv::Mat& v, const cv::Mat& Ix, const cv::Mat& Iy, const cv::Mat& It0,
                        double alpha, HornSchunckSolver solver) {
    const float alpha2 = float(alpha * alpha);
    // Jacobi reads the neighbours from the previous field (a copy); Gauss–Seidel from the field
    // being updated, so the pixels above and to the left are already new.
    const bool jacobi = solver == HornSchunckSolver::Jacobi;
    const cv::Mat uPrev = jacobi ? u.clone() : u, vPrev = jacobi ? v.clone() : v;
    double sumUpdate = 0;
    for (int y = 0; y < u.rows; ++y)
        for (int x = 0; x < u.cols; ++x) {
            const NeighbourMean m = neighbourMean(uPrev, vPrev, y, x);
            const float ix = Ix.at<float>(y, x), iy = Iy.at<float>(y, x);
            const float t = (ix * m.u + iy * m.v + It0.at<float>(y, x)) / (alpha2 * m.s + ix * ix + iy * iy);
            const float uNew = m.u - ix * t, vNew = m.v - iy * t;
            const float du = uNew - u.at<float>(y, x), dv = vNew - v.at<float>(y, x);
            sumUpdate += std::sqrt(du * du + dv * dv);
            u.at<float>(y, x) = uNew;
            v.at<float>(y, x) = vNew;
        }
    return sumUpdate / double(u.total());
}

double hornSchunckEnergy(const cv::Mat& u, const cv::Mat& v, const cv::Mat& Ix, const cv::Mat& Iy,
                         const cv::Mat& It0, double alpha) {
    double data = 0, smooth = 0;
    for (int y = 0; y < u.rows; ++y)
        for (int x = 0; x < u.cols; ++x) {
            const double r = double(Ix.at<float>(y, x)) * u.at<float>(y, x) + double(Iy.at<float>(y, x)) * v.at<float>(y, x) +
                             It0.at<float>(y, x);
            data += r * r;
            // Each pair once: from every pixel, only to its right, down-left, down and down-right neighbours.
            for (const Neighbour& n : {kNeighbours[1], kNeighbours[3], kNeighbours[6], kNeighbours[7]}) {
                const int xx = x + n.dx, yy = y + n.dy;
                if (xx < 0 || yy >= u.rows || xx >= u.cols) continue;
                const double du = u.at<float>(y, x) - u.at<float>(yy, xx), dv = v.at<float>(y, x) - v.at<float>(yy, xx);
                smooth += n.weight * (du * du + dv * dv);
            }
        }
    return (data + alpha * alpha * smooth) / double(u.total());
}

namespace {

cv::Mat merged(const cv::Mat& u, const cv::Mat& v) {
    cv::Mat w;
    cv::merge(std::vector<cv::Mat>{u, v}, w);
    return w;
}

// Share of each pixel's update that comes from its own data rather than its neighbours:
// |∇I|² / (|∇I|² + α²) (with s = 1, its value inside the image).
cv::Mat dataWeight(const Derivatives& d, double alpha) {
    const cv::Mat g2 = d.Ix.mul(d.Ix) + d.Iy.mul(d.Iy);
    return g2 / (g2 + alpha * alpha);
}

// Where a level reports to: cancellation, progress bar, partial pictures.
struct Reporting {
    std::stop_token stop;
    const ProgressFn& progress;
    const FlowCallback& onStep;
    cv::Size full;  // frame size, for the onStep pictures
};

// Steps 3 to 5 on one pyramid level l (of L): `warps` linearisations, the sweeps split between them.
void solveLevel(const cv::Mat& I0, const cv::Mat& I1, int l, int L, const HornSchunckParams& p, cv::Mat& u,
                cv::Mat& v, HornSchunckResult& res, const Reporting& out) {
    const int warps = std::clamp(p.warps, 1, p.iterations), every = std::max(1, p.reportEvery);
    const double toFull = double(1 << l);
    const std::string level = "level " + std::to_string(L - l) + " / " + std::to_string(L);
    int sweep = 0;
    for (int k = 0; k < warps; ++k) {
        // 3. Linearise around w0 = (u, v): r = Ix·(u − u0) + Iy·(v − v0) + It = Ix·u + Iy·v + It0.
        const Derivatives d = linearise(I0, I1, merged(u, v));
        const cv::Mat It0 = d.It - d.Ix.mul(u) - d.Iy.mul(v);
        if (l == 0 && k == 0) {
            res.dataWeight = dataWeight(d, p.alpha);
            res.energy.push_back(hornSchunckEnergy(u, v, d.Ix, d.Iy, It0, p.alpha));
        }
        const int sweeps = (p.iterations * (k + 1)) / warps - (p.iterations * k) / warps;
        for (int i = 0; i < sweeps; ++i, ++sweep, ++res.sweeps) {
            checkStop(out.stop);
            res.meanUpdate.push_back(hornSchunckSweep(u, v, d.Ix, d.Iy, It0, p.alpha, p.solver) * toFull);  // 4.
            if (l == 0) res.energy.push_back(hornSchunckEnergy(u, v, d.Ix, d.Iy, It0, p.alpha));          // 5.
            if ((sweep + 1) % every != 0 && sweep + 1 != p.iterations) continue;
            const std::string stage = level + ", sweep " + std::to_string(sweep + 1);
            report(out.progress, double((L - 1 - l) * p.iterations + sweep + 1) / (L * p.iterations), stage);
            if (out.onStep) out.onStep(upsampleFlow(merged(u, v), out.full, float(toFull)), stage);
        }
    }
}

}  // namespace

HornSchunckResult hornSchunck(const cv::Mat& frame0, const cv::Mat& frame1, const HornSchunckParams& p,
                              std::stop_token stop, const ProgressFn& progress, const FlowCallback& onStep) {
    CV_Assert(frame0.size() == frame1.size() && p.alpha > 0 && p.levels >= 1 && p.iterations >= 1);
    // 1. Pyramids of both frames.
    const auto pyr0 = buildPyramid(prepareFrame(frame0, p.presmooth), p.levels);
    const auto pyr1 = buildPyramid(prepareFrame(frame1, p.presmooth), int(pyr0.size()));
    const int L = int(pyr0.size());

    HornSchunckResult res;
    res.levelsUsed = L;
    cv::Mat u = cv::Mat::zeros(pyr0.back().size(), CV_32F), v = u.clone();
    for (int l = L - 1; l >= 0; --l) {
        // 2. Start from the coarser level's answer: twice the vectors, on a grid twice as fine.
        if (l < L - 1) {
            std::vector<cv::Mat> uv;
            cv::split(upsampleFlow(merged(u, v), pyr0[size_t(l)].size(), 2.f), uv);
            u = uv[0];
            v = uv[1];
        }
        solveLevel(pyr0[size_t(l)], pyr1[size_t(l)], l, L, p, u, v, res, {stop, progress, onStep, frame0.size()});
    }
    res.flow = merged(u, v);
    report(progress, 1.0, "done");
    return res;
}

}  // namespace vl::flow
