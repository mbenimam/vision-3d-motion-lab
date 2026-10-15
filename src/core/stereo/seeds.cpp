// Stereo by seed propagation. Course: "3D Vision and Motion", part 6.
//
// Block matching decides each pixel alone, so textureless areas get random answers. Here a pixel is
// only matched when it is sure, or when a sure neighbour vouches for it:
//   1. scoreVolume   ZNCC(x, y, d) for every pixel and disparity (block matching's cost volume, as
//                    a score: ZNCC = 1 − cost, in [−1, 1], 1 = same shape)
//   2. selectSeeds   pixels whose best ZNCC is at least the seed threshold and that pass the
//                    left-right check; optionally only the best one per spacing × spacing cell
//   3. grow          a priority queue of reached pixels, highest score first. Pop P (disparity dP);
//                    each 4-neighbour Q not reached yet tries only dP − 1, dP, dP + 1 (bestNearby):
//                    surfaces are smooth, so a neighbour's disparity is close. Q is accepted, and
//                    queued, if its best score is still above the growth floor.
// The map grows from the reliable regions outwards and stops at textureless areas, occlusions and
// depth edges, where no nearby disparity correlates well.
//
// Cost: the volume is O(W·H·D) with integral images; the growth O(W·H·log(W·H)) for the queue.

#include "core/stereo/seeds.hpp"

#include <cmath>
#include <limits>
#include <queue>
#include <string>

#include "core/stereo/block_matching.hpp"

namespace vl::stereo {
namespace {

constexpr float kNoScore = -std::numeric_limits<float>::infinity();

// Step 1. The scores, and block matching's winners after its left-right check (NaN where it failed).
struct ScoreVolume {
    std::vector<cv::Mat> cost;  // cost[k] = 1 − ZNCC at disparity minDisparity + k
    cv::Mat winners;            // CV_32F
    int minDisparity = 0;

    // ZNCC of left (x, y) with right (x − d, y); kNoScore outside the range or the image.
    float score(int x, int y, int d) const {
        const int k = d - minDisparity;
        if (k < 0 || k >= int(cost.size())) return kNoScore;
        const float c = cost[std::size_t(k)].at<float>(y, x);
        return std::isfinite(c) ? 1.f - c : kNoScore;
    }
};

ScoreVolume scoreVolume(const cv::Mat& left, const cv::Mat& right, const SeedParams& p, std::stop_token stop,
                        const ProgressFn& progress) {
    BlockMatchParams bp;
    bp.minDisparity = p.minDisparity;
    bp.maxDisparity = p.maxDisparity;
    bp.radius = p.radius;
    bp.cost = Cost::ZNCC;
    bp.leftRightCheck = p.leftRightCheck;
    bp.subpixel = false;
    bp.keepVolume = true;
    auto halfway = [&](double f, const std::string& m) { report(progress, 0.5 * f, m); };
    BlockMatchResult bm = blockMatch(left, right, bp, stop, halfway);
    return {std::move(bm.volume), bm.disparity, p.minDisparity};
}

// A match for pixel (x, y) at disparity d; the queue pops the highest score first.
struct Candidate {
    float score;
    int x, y, d;
    bool operator<(const Candidate& other) const { return score < other.score; }
};

// The result maps and the queue, shared by steps 2 and 3.
struct Growth {
    SeedResult res;
    std::priority_queue<Candidate> queue;

    void accept(const Candidate& c, int order) {
        res.disparity.at<float>(c.y, c.x) = float(c.d);
        res.score.at<float>(c.y, c.x) = c.score;
        res.order.at<int>(c.y, c.x) = order;
        queue.push(c);
    }
    bool reached(int x, int y) const { return res.order.at<int>(y, x) >= 0; }
};

// Step 2. In each cell, the confident winner with the highest score becomes a seed.
void selectSeeds(const ScoreVolume& v, const SeedParams& p, Growth& g) {
    const int W = v.winners.cols, H = v.winners.rows, cell = std::max(1, p.seedSpacing);
    for (int cy = 0; cy < H; cy += cell)
        for (int cx = 0; cx < W; cx += cell) {
            Candidate best{kNoScore, -1, -1, 0};
            for (int y = cy; y < std::min(H, cy + cell); ++y)
                for (int x = cx; x < std::min(W, cx + cell); ++x) {
                    const float d = v.winners.at<float>(y, x);
                    if (!std::isfinite(d)) continue;  // rejected by the left-right check
                    const int di = int(std::lround(d));
                    const Candidate c{v.score(x, y, di), x, y, di};
                    if (c.score >= p.seedScore && c.score > best.score) best = c;
                }
            if (best.x < 0) continue;
            g.res.seeds.emplace_back(best.x, best.y);
            g.accept(best, 0);
        }
}

// The best of d − 1, d, d + 1 for pixel (x, y).
Candidate bestNearby(const ScoreVolume& v, int x, int y, int d) {
    Candidate best{kNoScore, x, y, d};
    for (int dd = d - 1; dd <= d + 1; ++dd)
        if (v.score(x, y, dd) > best.score) best = {v.score(x, y, dd), x, y, dd};
    return best;
}

// Step 3. Best-first growth. A neighbour refused now may still be reached later from another side.
void grow(const ScoreVolume& v, const SeedParams& p, Growth& g, std::stop_token stop, const ProgressFn& progress,
          const SeedObserver& observer, int every) {
    const int W = v.winners.cols, H = v.winners.rows;
    const int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
    while (!g.queue.empty()) {
        const Candidate P = g.queue.top();
        g.queue.pop();
        for (int n = 0; n < 4; ++n) {
            const int x = P.x + dx[n], y = P.y + dy[n];
            if (x < 0 || y < 0 || x >= W || y >= H || g.reached(x, y)) continue;
            const Candidate Q = bestNearby(v, x, y, P.d);
            if (Q.score < p.stopScore) continue;
            g.accept(Q, ++g.res.grown);
            if (g.res.grown % every != 0) continue;
            checkStop(stop);
            const int reached = g.res.grown + int(g.res.seeds.size());
            report(progress, 0.5 + 0.5 * reached / double(W * H), "growing: " + std::to_string(reached) + " pixels");
            if (observer) observer(g.res.disparity, g.res.order, reached);
        }
    }
}

}  // namespace

SeedResult propagateSeeds(const cv::Mat& left, const cv::Mat& right, const SeedParams& p, std::stop_token stop,
                          const ProgressFn& progress, const SeedObserver& observer, int every) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Growth g;
    g.res.disparity = cv::Mat(left.size(), CV_32F, cv::Scalar(nan));
    g.res.score = cv::Mat(left.size(), CV_32F, cv::Scalar(nan));
    g.res.order = cv::Mat(left.size(), CV_32S, cv::Scalar(-1));

    const ScoreVolume v = scoreVolume(left, right, p, stop, progress);  // 1.
    selectSeeds(v, p, g);                                                // 2.
    const int reportEvery = every > 0 ? every : std::max(1, int(left.total() / 20));
    grow(v, p, g, stop, progress, observer, reportEvery);                // 3.
    report(progress, 1.0, "done");
    return std::move(g.res);
}

}  // namespace vl::stereo
