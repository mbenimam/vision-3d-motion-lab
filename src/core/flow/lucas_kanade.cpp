#include "core/flow/lucas_kanade.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace vl::flow {

LucasKanadeResult lucasKanade(const cv::Mat& frame0, const cv::Mat& frame1, const LucasKanadeParams& p,
                              std::stop_token stop, const ProgressFn& progress, const FlowCallback& onStep) {
    CV_Assert(frame0.size() == frame1.size() && p.sigma > 0 && p.levels >= 1 && p.iterations >= 1);
    const auto pyr0 = gaussianPyramid(prepareFrame(frame0, p.presmooth), p.levels);
    const auto pyr1 = gaussianPyramid(prepareFrame(frame1, p.presmooth), int(pyr0.size()));
    const int L = int(pyr0.size());
    const float eps = float(p.tikhonov);

    LucasKanadeResult res;
    res.levelsUsed = L;
    cv::Mat w = cv::Mat::zeros(pyr0.back().size(), CV_32FC2);
    cv::Mat a, b, c, bx, by, outside;  // J = [a b; b c], right-hand side (bx, by)
    const int total = L * p.iterations;
    for (int l = L - 1; l >= 0; --l) {
        const cv::Mat& I0 = pyr0[size_t(l)];
        const cv::Mat& I1 = pyr1[size_t(l)];
        if (w.size() != I0.size()) w = resizeFlow(w, I0.size());
        const float toFull = float(frame0.cols) / float(I0.cols);  // level pixels -> full-resolution pixels
        for (int it = 0; it < p.iterations; ++it) {
            checkStop(stop);
            const int step = (L - 1 - l) * p.iterations + it;
            report(progress, double(step) / total,
                   "level " + std::to_string(L - l) + " / " + std::to_string(L) + ", iteration " + std::to_string(it + 1));

            // Linearise around the current flow: compare I0 with I1 warped back by w.
            const cv::Mat I1w = warpImage(I1, w, &outside);
            Derivatives d = derivatives(I0, I1w, true);
            d.Ix.setTo(0, outside);  // a pixel that left the frame says nothing
            d.Iy.setTo(0, outside);
            d.It.setTo(0, outside);

            // Each pixel x' of the window was linearised around its own flow w(x'), so the constraint it
            // gives on the window's (constant) flow W is  ∇I(x')·(W - w(x')) + It(x') = 0. Least squares
            // over the window: J·W = G * [∇I ∇Iᵀ w - ∇I It] =: r. Solving for W rather than for an
            // increment keeps the iterations exact when w varies inside the window (the classic increment
            // form would mix the flows of different pixels and drift). Five Gaussian sums per iteration.
            std::vector<cv::Mat> uv;
            cv::split(w, uv);
            const cv::Mat xx = d.Ix.mul(d.Ix), xy = d.Ix.mul(d.Iy), yy = d.Iy.mul(d.Iy);
            const auto window = [&](const cv::Mat& m, cv::Mat& out) {
                cv::GaussianBlur(m, out, cv::Size(), p.sigma, p.sigma, cv::BORDER_REPLICATE);
            };
            window(xx, a);
            window(xy, b);
            window(yy, c);
            window(xx.mul(uv[0]) + xy.mul(uv[1]) - d.Ix.mul(d.It), bx);
            window(xy.mul(uv[0]) + yy.mul(uv[1]) - d.Iy.mul(d.It), by);

            const bool last = l == 0 && it == p.iterations - 1;
            if (last) {
                res.minEigen.create(I0.size(), CV_32F);
                res.maxEigen.create(I0.size(), CV_32F);
                cv::merge(std::vector<cv::Mat>{a, b, c}, res.tensor);
            }
            double sumUpdate = 0;
            for (int y = 0; y < I0.rows; ++y) {
                const float *pa = a.ptr<float>(y), *pb = b.ptr<float>(y), *pc = c.ptr<float>(y);
                const float *pbx = bx.ptr<float>(y), *pby = by.ptr<float>(y);
                auto* pw = w.ptr<cv::Vec2f>(y);
                float* lo = last ? res.minEigen.ptr<float>(y) : nullptr;
                float* hi = last ? res.maxEigen.ptr<float>(y) : nullptr;
                for (int x = 0; x < I0.cols; ++x) {
                    // (J + εI)·W = r + ε·w: the small ε keeps the previous estimate where the window
                    // says nothing (flat area), and along an edge only the normal component changes.
                    const float A = pa[x] + eps, B = pb[x], C = pc[x] + eps;
                    const float det = A * C - B * B;
                    const float rx = pbx[x] + eps * pw[x][0], ry = pby[x] + eps * pw[x][1];
                    const float nu = (C * rx - B * ry) / det, nv = (A * ry - B * rx) / det;
                    const float du = nu - pw[x][0], dv = nv - pw[x][1];
                    pw[x][0] = nu;
                    pw[x][1] = nv;
                    sumUpdate += std::sqrt(du * du + dv * dv);
                    if (lo) {
                        // Eigenvalues of [a b; b c]: mean ± sqrt(((a - c)/2)² + b²).
                        const float m = 0.5f * (pa[x] + pc[x]);
                        const float r = std::sqrt(0.25f * (pa[x] - pc[x]) * (pa[x] - pc[x]) + pb[x] * pb[x]);
                        lo[x] = std::max(0.f, m - r);
                        hi[x] = m + r;
                    }
                }
            }
            res.meanUpdate.push_back(sumUpdate / double(I0.total()) * toFull);
            if (onStep)
                onStep(resizeFlow(w, frame0.size()),
                       "level " + std::to_string(L - l) + " / " + std::to_string(L) + ", iteration " + std::to_string(it + 1));
        }
    }

    if (p.minEigen > 0) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        w.setTo(cv::Scalar(nan, nan), res.minEigen < p.minEigen);
    }
    res.flow = w;
    report(progress, 1.0, "done");
    return res;
}

}  // namespace vl::flow
