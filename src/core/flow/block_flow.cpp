#include "core/flow/block_flow.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "core/common/image.hpp"

namespace vl::flow {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Mean cost of the block around (x, y) against frame 1 displaced by (dx, dy). Coordinates are
// clamped to the image, which is what the box filter with replicated borders does in the fast path.
float blockCost(const cv::Mat& I0, const cv::Mat& I1, int x, int y, int dx, int dy, int r, BlockCost c) {
    const int W = I0.cols, H = I0.rows;
    double s = 0;
    for (int j = -r; j <= r; ++j) {
        const int yy = std::clamp(y + j, 0, H - 1), y1 = std::clamp(yy + dy, 0, H - 1);
        const float* a = I0.ptr<float>(yy);
        const float* b = I1.ptr<float>(y1);
        for (int i = -r; i <= r; ++i) {
            const int xx = std::clamp(x + i, 0, W - 1), x1 = std::clamp(xx + dx, 0, W - 1);
            const float d = a[xx] - b[x1];
            s += c == BlockCost::SAD ? std::abs(d) : d * d;
        }
    }
    return float(s / double((2 * r + 1) * (2 * r + 1)));
}

// Vertex of the parabola through (-1, a), (0, b), (1, c), clamped to half a pixel.
float parabola(float a, float b, float c) {
    const float den = a - 2 * b + c;
    if (!(den > 1e-6f) || !std::isfinite(a) || !std::isfinite(c)) return 0.f;
    return std::clamp(0.5f * (a - c) / den, -0.5f, 0.5f);
}

}  // namespace

BlockFlowResult blockMatchFlow(const cv::Mat& frame0, const cv::Mat& frame1, const BlockFlowParams& p,
                               std::stop_token stop, const ProgressFn& progress) {
    CV_Assert(frame0.size() == frame1.size() && p.radius >= 0 && p.search >= 0);
    const cv::Mat I0 = toGrayF(frame0), I1 = toGrayF(frame1);
    const int W = I0.cols, H = I0.rows, S = p.search, r = p.radius;
    cv::Mat I1pad;  // frame 1 with S replicated pixels around it, so every shift is a plain sub-image
    cv::copyMakeBorder(I1, I1pad, S, S, S, S, cv::BORDER_REPLICATE);

    // Visit displacements from the smallest to the largest: on ties (flat areas) the smallest
    // motion wins, which is the least surprising answer.
    std::vector<cv::Point> shifts;
    for (int dy = -S; dy <= S; ++dy)
        for (int dx = -S; dx <= S; ++dx) shifts.emplace_back(dx, dy);
    std::stable_sort(shifts.begin(), shifts.end(),
                     [](cv::Point a, cv::Point b) { return a.x * a.x + a.y * a.y < b.x * b.x + b.y * b.y; });

    BlockFlowResult res;
    res.cost.create(I0.size(), CV_32F);
    res.cost.setTo(kInf);
    cv::Mat best(I0.size(), CV_32SC2, cv::Scalar(0, 0));
    cv::Mat diff, c;
    const cv::Size block(2 * r + 1, 2 * r + 1);
    for (size_t k = 0; k < shifts.size(); ++k) {
        if (k % 16 == 0) {
            checkStop(stop);
            report(progress, 0.9 * double(k) / double(shifts.size()), "displacement " + std::to_string(k + 1) + " / " +
                                                                          std::to_string(shifts.size()));
        }
        const cv::Point d = shifts[k];
        const cv::Mat I1s = I1pad(cv::Rect(S + d.x, S + d.y, W, H));  // I1s(x) = I1(x + d)
        cv::absdiff(I0, I1s, diff);
        if (p.cost == BlockCost::SSD) diff = diff.mul(diff);
        cv::boxFilter(diff, c, CV_32F, block, cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
        // A displacement that sends the pixel itself out of frame 1 is not a candidate.
        const int x0 = std::max(0, -d.x), x1 = std::min(W, W - d.x), y0 = std::max(0, -d.y), y1 = std::min(H, H - d.y);
        for (int y = y0; y < y1; ++y) {
            const float* pc = c.ptr<float>(y);
            float* pb = res.cost.ptr<float>(y);
            auto* pk = best.ptr<cv::Vec2i>(y);
            for (int x = x0; x < x1; ++x)
                if (pc[x] < pb[x]) {
                    pb[x] = pc[x];
                    pk[x] = {d.x, d.y};
                }
        }
    }
    res.evaluations = long(W) * H * long(shifts.size());

    res.flow.create(I0.size(), CV_32FC2);
    for (int y = 0; y < H; ++y) {
        const auto* pk = best.ptr<cv::Vec2i>(y);
        const float* pb = res.cost.ptr<float>(y);
        auto* f = res.flow.ptr<cv::Vec2f>(y);
        for (int x = 0; x < W; ++x) {
            const int dx = pk[x][0], dy = pk[x][1];
            float u = float(dx), v = float(dy);
            if (p.subpixel) {
                // Sub-pixel: one parabola along x, one along y, through the neighbouring costs.
                // 4 extra block costs per pixel: O(W·H·(2r+1)²).
                const auto cost = [&](int ex, int ey) {
                    const int tx = x + dx + ex, ty = y + dy + ey;
                    if (std::abs(dx + ex) > S || std::abs(dy + ey) > S || tx < 0 || ty < 0 || tx >= W || ty >= H) return kInf;
                    return blockCost(I0, I1, x, y, dx + ex, dy + ey, r, p.cost);
                };
                u += parabola(cost(-1, 0), pb[x], cost(1, 0));
                v += parabola(cost(0, -1), pb[x], cost(0, 1));
            }
            f[x] = {u, v};
        }
        if (y % 32 == 0) checkStop(stop);
    }
    report(progress, 1.0, "done");
    return res;
}

cv::Mat blockCostSurface(const cv::Mat& I0, const cv::Mat& I1, cv::Point p, const BlockFlowParams& params) {
    CV_Assert(I0.type() == CV_32F && I1.type() == CV_32F && I0.size() == I1.size());
    const int S = params.search;
    cv::Mat out(2 * S + 1, 2 * S + 1, CV_32F);
    for (int dy = -S; dy <= S; ++dy)
        for (int dx = -S; dx <= S; ++dx) {
            const int tx = p.x + dx, ty = p.y + dy;
            out.at<float>(dy + S, dx + S) = tx < 0 || ty < 0 || tx >= I0.cols || ty >= I0.rows
                                                ? kInf
                                                : blockCost(I0, I1, p.x, p.y, dx, dy, params.radius, params.cost);
        }
    return out;
}

}  // namespace vl::flow
