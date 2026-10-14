#include "core/flow/horn_schunck.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include <opencv2/imgproc.hpp>

namespace vl::flow {

namespace {

constexpr float kEdge = 1.f / 6.f, kDiag = 1.f / 12.f;

// Weighted neighbour sums of u and v at (x, y) and the total weight of the neighbours that exist.
struct Neighbours {
    float su = 0, sv = 0, s = 0;
};

inline Neighbours neighbours(const float* const ur[3], const float* const vr[3], int x, int W) {
    Neighbours n;
    const bool left = x > 0, right = x < W - 1;
    for (int j = 0; j < 3; ++j) {
        if (!ur[j]) continue;
        const float wc = j == 1 ? 0.f : kEdge, ws = j == 1 ? kEdge : kDiag;  // centre column / side columns
        if (wc > 0) {
            n.su += wc * ur[j][x];
            n.sv += wc * vr[j][x];
            n.s += wc;
        }
        if (left) {
            n.su += ws * ur[j][x - 1];
            n.sv += ws * vr[j][x - 1];
            n.s += ws;
        }
        if (right) {
            n.su += ws * ur[j][x + 1];
            n.sv += ws * vr[j][x + 1];
            n.s += ws;
        }
    }
    return n;
}

template <class T>
void rows(const cv::Mat& m, int y, const T* out[3]) {
    out[0] = y > 0 ? m.ptr<T>(y - 1) : nullptr;
    out[1] = m.ptr<T>(y);
    out[2] = y < m.rows - 1 ? m.ptr<T>(y + 1) : nullptr;
}

}  // namespace

double hornSchunckSweep(cv::Mat& u, cv::Mat& v, const cv::Mat& Ix, const cv::Mat& Iy, const cv::Mat& It0,
                        double alpha, HornSchunckSolver solver) {
    const int H = u.rows, W = u.cols;
    const float a2 = float(alpha * alpha);
    // Jacobi reads the previous field from a copy; Gauss–Seidel reads and writes the same buffers.
    cv::Mat su, sv;
    if (solver == HornSchunckSolver::Jacobi) {
        su = u.clone();
        sv = v.clone();
    } else {
        su = u;
        sv = v;
    }
    double sumUpdate = 0;
    const auto update = [&](float mu, float mv, float s, float ix, float iy, float it, float& pu, float& pv) {
        const float t = (ix * mu + iy * mv + it) / (a2 * s + ix * ix + iy * iy);
        const float nu = mu - ix * t, nv = mv - iy * t;
        sumUpdate += std::sqrt((nu - pu) * (nu - pu) + (nv - pv) * (nv - pv));
        pu = nu;
        pv = nv;
    };
    for (int y = 0; y < H; ++y) {
        const float *ur[3], *vr[3];
        rows(su, y, ur);
        rows(sv, y, vr);
        const float *ix = Ix.ptr<float>(y), *iy = Iy.ptr<float>(y), *it = It0.ptr<float>(y);
        float* pu = u.ptr<float>(y);
        float* pv = v.ptr<float>(y);
        const bool inner = ur[0] && ur[2];
        for (int x = 0; x < W; ++x) {
            if (inner && x > 0 && x < W - 1) {
                // Interior: all eight neighbours exist, weights sum to 1 (the hot path, no branches).
                const float mu = kEdge * (ur[0][x] + ur[2][x] + ur[1][x - 1] + ur[1][x + 1]) +
                                 kDiag * (ur[0][x - 1] + ur[0][x + 1] + ur[2][x - 1] + ur[2][x + 1]);
                const float mv = kEdge * (vr[0][x] + vr[2][x] + vr[1][x - 1] + vr[1][x + 1]) +
                                 kDiag * (vr[0][x - 1] + vr[0][x + 1] + vr[2][x - 1] + vr[2][x + 1]);
                update(mu, mv, 1.f, ix[x], iy[x], it[x], pu[x], pv[x]);
            } else {
                const Neighbours n = neighbours(ur, vr, x, W);
                update(n.su / n.s, n.sv / n.s, n.s, ix[x], iy[x], it[x], pu[x], pv[x]);
            }
        }
    }
    return sumUpdate / double(u.total());
}

double hornSchunckEnergy(const cv::Mat& u, const cv::Mat& v, const cv::Mat& Ix, const cv::Mat& Iy,
                         const cv::Mat& It0, double alpha) {
    // Data term at every pixel, smoothness once per unordered neighbour pair: right, down, and the
    // two diagonals below.
    const int H = u.rows, W = u.cols;
    double data = 0, smooth = 0;
    for (int y = 0; y < H; ++y) {
        const float *pu = u.ptr<float>(y), *pv = v.ptr<float>(y);
        const float *nu = y + 1 < H ? u.ptr<float>(y + 1) : nullptr, *nv = y + 1 < H ? v.ptr<float>(y + 1) : nullptr;
        const float *ix = Ix.ptr<float>(y), *iy = Iy.ptr<float>(y), *it = It0.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            const double r = double(ix[x]) * pu[x] + double(iy[x]) * pv[x] + it[x];
            data += r * r;
            const auto pair = [&](float w, float u2, float v2) {
                smooth += w * (double(pu[x] - u2) * (pu[x] - u2) + double(pv[x] - v2) * (pv[x] - v2));
            };
            if (x + 1 < W) pair(kEdge, pu[x + 1], pv[x + 1]);
            if (nu) {
                pair(kEdge, nu[x], nv[x]);
                if (x + 1 < W) pair(kDiag, nu[x + 1], nv[x + 1]);
                if (x > 0) pair(kDiag, nu[x - 1], nv[x - 1]);
            }
        }
    }
    return (data + alpha * alpha * smooth) / double(u.total());
}

HornSchunckResult hornSchunck(const cv::Mat& frame0, const cv::Mat& frame1, const HornSchunckParams& p,
                              std::stop_token stop, const ProgressFn& progress, const FlowCallback& onStep) {
    CV_Assert(frame0.size() == frame1.size() && p.alpha > 0 && p.levels >= 1 && p.iterations >= 1);
    const auto pyr0 = gaussianPyramid(prepareFrame(frame0, p.presmooth), p.levels);
    const auto pyr1 = gaussianPyramid(prepareFrame(frame1, p.presmooth), int(pyr0.size()));
    const int L = int(pyr0.size());
    const int warps = std::clamp(p.warps, 1, p.iterations);
    const int every = std::max(1, p.reportEvery);

    HornSchunckResult res;
    res.levelsUsed = L;
    cv::Mat u = cv::Mat::zeros(pyr0.back().size(), CV_32F), v = u.clone();
    cv::Mat outside;
    const int total = L * p.iterations;
    for (int l = L - 1; l >= 0; --l) {
        const cv::Mat& I0 = pyr0[size_t(l)];
        const cv::Mat& I1 = pyr1[size_t(l)];
        if (u.size() != I0.size()) {
            cv::Mat uv;
            cv::merge(std::vector<cv::Mat>{u, v}, uv);
            uv = resizeFlow(uv, I0.size());
            std::vector<cv::Mat> ch;
            cv::split(uv, ch);
            u = ch[0];
            v = ch[1];
        }
        const double toFull = double(frame0.cols) / double(I0.cols);
        const std::string level = "level " + std::to_string(L - l) + " / " + std::to_string(L);
        int sweep = 0;
        for (int k = 0; k < warps; ++k) {
            // Linearise around the current field w0: r = Ix·(u - u0) + Iy·(v - v0) + (I1(x + w0) - I0)
            //                                          = Ix·u + Iy·v + It0.
            cv::Mat w0;
            cv::merge(std::vector<cv::Mat>{u, v}, w0);
            const cv::Mat I1w = warpImage(I1, w0, &outside);
            Derivatives d = derivatives(I0, I1w, true);
            d.Ix.setTo(0, outside);
            d.Iy.setTo(0, outside);
            d.It.setTo(0, outside);
            const cv::Mat It0 = d.It - d.Ix.mul(u) - d.Iy.mul(v);
            if (l == 0 && k == 0) {
                // Share of each pixel's update that comes from its own data rather than its neighbours
                // (s = 1 inside the image; the border's smaller s is ignored here).
                const cv::Mat g2 = d.Ix.mul(d.Ix) + d.Iy.mul(d.Iy);
                res.dataWeight = g2 / (g2 + p.alpha * p.alpha);
                res.energy.push_back(hornSchunckEnergy(u, v, d.Ix, d.Iy, It0, p.alpha));
            }

            const int n = (p.iterations * (k + 1)) / warps - (p.iterations * k) / warps;
            for (int i = 0; i < n; ++i, ++sweep) {
                checkStop(stop);
                const double upd = hornSchunckSweep(u, v, d.Ix, d.Iy, It0, p.alpha, p.solver);
                res.meanUpdate.push_back(upd * toFull);
                if (l == 0) res.energy.push_back(hornSchunckEnergy(u, v, d.Ix, d.Iy, It0, p.alpha));
                ++res.sweeps;
                if ((sweep + 1) % every == 0 || sweep + 1 == p.iterations) {
                    const int done = (L - 1 - l) * p.iterations + sweep + 1;
                    const std::string stage = level + ", sweep " + std::to_string(sweep + 1);
                    report(progress, double(done) / total, stage);
                    if (onStep) {
                        cv::Mat w;
                        cv::merge(std::vector<cv::Mat>{u, v}, w);
                        onStep(resizeFlow(w, frame0.size()), stage);
                    }
                }
            }
        }
    }
    cv::merge(std::vector<cv::Mat>{u, v}, res.flow);
    report(progress, 1.0, "done");
    return res;
}

}  // namespace vl::flow
