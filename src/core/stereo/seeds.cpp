#include "core/stereo/seeds.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <string>

#include "core/stereo/block_matching.hpp"

namespace vl::stereo {

SeedResult propagateSeeds(const cv::Mat& left, const cv::Mat& right, const SeedParams& p, std::stop_token stop,
                          const ProgressFn& progress, const SeedObserver& observer, int every) {
    // Scores for every (pixel, disparity) from the block matcher: volume[k] = 1 - ZNCC at d = min + k.
    BlockMatchParams bp;
    bp.minDisparity = p.minDisparity;
    bp.maxDisparity = p.maxDisparity;
    bp.radius = p.radius;
    bp.cost = Cost::ZNCC;
    bp.leftRightCheck = p.leftRightCheck;
    bp.subpixel = false;
    bp.keepVolume = true;
    const auto bm = blockMatch(left, right, bp, stop, [&](double f, const std::string& m) { report(progress, 0.5 * f, m); });
    const int W = left.cols, H = left.rows, D = int(bm.volume.size());
    auto score = [&](int x, int y, int d) -> float {
        const int k = d - p.minDisparity;
        if (k < 0 || k >= D) return -std::numeric_limits<float>::infinity();
        const float c = bm.volume[std::size_t(k)].at<float>(y, x);
        return std::isfinite(c) ? 1.f - c : -std::numeric_limits<float>::infinity();
    };

    SeedResult res;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    res.disparity = cv::Mat(left.size(), CV_32F, cv::Scalar(nan));
    res.score = cv::Mat(left.size(), CV_32F, cv::Scalar(nan));
    res.order = cv::Mat(left.size(), CV_32S, cv::Scalar(-1));

    struct Item {
        float score;
        int x, y;
        bool operator<(const Item& o) const { return score < o.score; }  // max-heap on the score
    };
    std::priority_queue<Item> queue;
    auto accept = [&](int x, int y, int d, float s, int order) {
        res.disparity.at<float>(y, x) = float(d);
        res.score.at<float>(y, x) = s;
        res.order.at<int>(y, x) = order;
        queue.push({s, x, y});
    };

    // Seeds: confident winners of the left-to-right search that the right-to-left search confirms,
    // thinned to the best candidate of each cell when seedSpacing > 1.
    const int cell = std::max(1, p.seedSpacing);
    for (int cy = 0; cy < H; cy += cell)
        for (int cx = 0; cx < W; cx += cell) {
            float bestS = -1;
            int bx = -1, by = -1, bd = 0;
            for (int y = cy; y < std::min(H, cy + cell); ++y)
                for (int x = cx; x < std::min(W, cx + cell); ++x) {
                    const float d = bm.disparity.at<float>(y, x);  // NaN where the L-R check failed
                    if (!std::isfinite(d)) continue;
                    const int di = int(std::lround(d));
                    const float s = score(x, y, di);
                    if (s >= p.seedScore && s > bestS) {
                        bestS = s;
                        bx = x;
                        by = y;
                        bd = di;
                    }
                }
            if (bx < 0) continue;
            res.seeds.emplace_back(bx, by);
            accept(bx, by, bd, bestS, 0);
        }

    // Best-first growth.
    const int reportEvery = every > 0 ? every : std::max(1, W * H / 20);
    const int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
    while (!queue.empty()) {
        const Item it = queue.top();
        queue.pop();
        const int dP = int(res.disparity.at<float>(it.y, it.x));
        for (int n = 0; n < 4; ++n) {
            const int x = it.x + dx[n], y = it.y + dy[n];
            if (x < 0 || y < 0 || x >= W || y >= H || res.order.at<int>(y, x) >= 0) continue;
            int best = dP;
            float bs = -std::numeric_limits<float>::infinity();
            for (int d = dP - 1; d <= dP + 1; ++d) {
                const float s = score(x, y, d);
                if (s > bs) {
                    bs = s;
                    best = d;
                }
            }
            if (bs < p.stopScore) continue;  // may still be reached later from another neighbour
            accept(x, y, best, bs, ++res.grown);
            if (res.grown % reportEvery == 0) {
                checkStop(stop);
                const int reached = res.grown + int(res.seeds.size());
                report(progress, 0.5 + 0.5 * reached / double(W * H), "growing: " + std::to_string(reached) + " pixels");
                if (observer) observer(res.disparity, res.order, reached);
            }
        }
    }
    report(progress, 1.0, "done");
    return res;
}

}  // namespace vl::stereo
