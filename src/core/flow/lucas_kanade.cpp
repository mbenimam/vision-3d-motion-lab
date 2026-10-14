// Dense pyramidal Lucas–Kanade. Course: "3D Vision and Motion", part 11.
//
// The method, one function per step:
//   1. buildPyramid        both frames, halved levels − 1 times (flow_common)
//   2. upsampleFlow        coarsest level first; each level starts from the coarser answer, doubled
//   3. linearise           warp I1 by the current flow w, take Ix, Iy, It (flow_common): the constraint is
//                          now about the small remainder only
//   4. windowSums          the structure tensor J and the right-hand side, summed over a Gaussian window
//   5. solveWindows        the 2×2 normal equations at every pixel, in closed form
//   6. tensorEigenvalues   λ1, λ2 of J at the finest level: how well each window pins the motion down
//   7. rejectWeakWindows   optional: no estimate where λ2 is small (Bouguet's reliability test)
// Steps 3 to 5 repeat `iterations` times per level (Gauss–Newton on brightness constancy).
//
// Cost: O(levels · iterations · W·H·σ): the window sums are separable Gaussian blurs.

#include "core/flow/lucas_kanade.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "core/common/filters.hpp"

namespace vl::flow {
namespace {

// J = [a b; b c] and the right-hand side r = (r1, r2) of the normal equations, per pixel.
struct WindowSystem {
    cv::Mat a, b, c, r1, r2;  // CV_32F
};

// Step 4. Each pixel x' of the window was linearised around its own flow w(x') = (u', v'), so the
// constraint it puts on the window's flow W = (U, V) is
//     Ix·(U − u') + Iy·(V − v') + It = 0.
// Least squares over the window, E(W) = Σ G(x − x')·(Ix·(U − u') + Iy·(V − v') + It)², and
// ∂E/∂U = ∂E/∂V = 0 give J·W = r with
//     J = G * [Ix²  IxIy; IxIy  Iy²],   r = G * [Ix²·u + IxIy·v − Ix·It;  IxIy·u + Iy²·v − Iy·It].
// Solving for W itself rather than for an increment stays exact when w varies inside the window
// (the increment form would mix the flows of different pixels and drift).
WindowSystem windowSums(const Derivatives& d, const cv::Mat& w, double sigma) {
    std::vector<cv::Mat> uv;
    cv::split(w, uv);
    const cv::Mat& u = uv[0];
    const cv::Mat& v = uv[1];
    const cv::Mat IxIx = d.Ix.mul(d.Ix), IxIy = d.Ix.mul(d.Iy), IyIy = d.Iy.mul(d.Iy);
    WindowSystem s;
    s.a = vl::gaussianBlur(IxIx, sigma);
    s.b = vl::gaussianBlur(IxIy, sigma);
    s.c = vl::gaussianBlur(IyIy, sigma);
    s.r1 = vl::gaussianBlur(IxIx.mul(u) + IxIy.mul(v) - d.Ix.mul(d.It), sigma);
    s.r2 = vl::gaussianBlur(IxIy.mul(u) + IyIy.mul(v) - d.Iy.mul(d.It), sigma);
    return s;
}

// Step 5. At every pixel, (J + εI)·W = r + ε·w solved by Cramer's rule:
//     det = (a + ε)(c + ε) − b²,   U = ((c + ε)·r1' − b·r2') / det,   V = ((a + ε)·r2' − b·r1') / det
// with r' = r + ε·w. The small ε keeps the previous estimate where the window says nothing (flat
// area); along an edge only the component across the edge changes. Returns the mean |Δw|.
double solveWindows(const WindowSystem& s, float eps, cv::Mat& w) {
    double sumUpdate = 0;
    for (int y = 0; y < w.rows; ++y)
        for (int x = 0; x < w.cols; ++x) {
            const cv::Vec2f old = w.at<cv::Vec2f>(y, x);
            const float a = s.a.at<float>(y, x) + eps, b = s.b.at<float>(y, x), c = s.c.at<float>(y, x) + eps;
            const float r1 = s.r1.at<float>(y, x) + eps * old[0], r2 = s.r2.at<float>(y, x) + eps * old[1];
            const float det = a * c - b * b;
            const cv::Vec2f W((c * r1 - b * r2) / det, (a * r2 - b * r1) / det);
            w.at<cv::Vec2f>(y, x) = W;
            sumUpdate += cv::norm(W - old);
        }
    return sumUpdate / double(w.total());
}

// Step 6. The eigenvalues of the symmetric J = [a b; b c]:
//     λ1,2 = (a + c)/2 ± √(((a − c)/2)² + b²)
void tensorEigenvalues(const WindowSystem& s, cv::Mat& lambda1, cv::Mat& lambda2) {
    lambda1.create(s.a.size(), CV_32F);
    lambda2.create(s.a.size(), CV_32F);
    for (int y = 0; y < s.a.rows; ++y)
        for (int x = 0; x < s.a.cols; ++x) {
            const float a = s.a.at<float>(y, x), b = s.b.at<float>(y, x), c = s.c.at<float>(y, x);
            const float mean = 0.5f * (a + c);
            const float radius = std::sqrt(0.25f * (a - c) * (a - c) + b * b);
            lambda1.at<float>(y, x) = mean + radius;
            lambda2.at<float>(y, x) = std::max(0.f, mean - radius);  // ≥ 0 in theory; rounding can dip below
        }
}

// Step 7. A window is reliable when even its weakest direction is constrained: λ2 ≥ threshold.
void rejectWeakWindows(cv::Mat& w, const cv::Mat& lambda2, double threshold) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int y = 0; y < w.rows; ++y)
        for (int x = 0; x < w.cols; ++x)
            if (lambda2.at<float>(y, x) < threshold) w.at<cv::Vec2f>(y, x) = {nan, nan};
}

}  // namespace

LucasKanadeResult lucasKanade(const cv::Mat& frame0, const cv::Mat& frame1, const LucasKanadeParams& p,
                              std::stop_token stop, const ProgressFn& progress, const FlowCallback& onStep) {
    CV_Assert(frame0.size() == frame1.size() && p.sigma > 0 && p.levels >= 1 && p.iterations >= 1);
    // 1. Pyramids of both frames.
    const auto pyr0 = buildPyramid(prepareFrame(frame0, p.presmooth), p.levels);
    const auto pyr1 = buildPyramid(prepareFrame(frame1, p.presmooth), int(pyr0.size()));
    const int L = int(pyr0.size());

    LucasKanadeResult res;
    res.levelsUsed = L;
    cv::Mat w = cv::Mat::zeros(pyr0.back().size(), CV_32FC2);
    WindowSystem system;
    for (int l = L - 1; l >= 0; --l) {
        const cv::Mat& I0 = pyr0[size_t(l)];
        const cv::Mat& I1 = pyr1[size_t(l)];
        // 2. Start from the coarser level's answer: twice the vectors, on a grid twice as fine.
        if (l < L - 1) w = upsampleFlow(w, I0.size(), 2.f);
        const float toFull = float(1 << l);  // one pixel of level l is 2^l pixels of the frame
        for (int it = 0; it < p.iterations; ++it) {
            checkStop(stop);
            const std::string stage =
                "level " + std::to_string(L - l) + " / " + std::to_string(L) + ", iteration " + std::to_string(it + 1);
            report(progress, double((L - 1 - l) * p.iterations + it) / (L * p.iterations), stage);
            const Derivatives d = linearise(I0, I1, w);                         // 3.
            system = windowSums(d, w, p.sigma);                                 // 4.
            res.meanUpdate.push_back(solveWindows(system, float(p.tikhonov), w) * toFull);  // 5.
            if (onStep) onStep(l == 0 ? w.clone() : upsampleFlow(w, frame0.size(), toFull), stage);
        }
    }

    // 6. The confidence of the last system solved, at full resolution.
    tensorEigenvalues(system, res.maxEigen, res.minEigen);
    cv::merge(std::vector<cv::Mat>{system.a, system.b, system.c}, res.tensor);
    // 7. Optional reliability test.
    if (p.minEigen > 0) rejectWeakWindows(w, res.minEigen, p.minEigen);
    res.flow = w;
    report(progress, 1.0, "done");
    return res;
}

}  // namespace vl::flow
