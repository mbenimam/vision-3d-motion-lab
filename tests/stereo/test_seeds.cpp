// Seed propagation: grows over most of a textured scene with few errors, in best-first order, each
// new pixel touching an earlier one at a disparity at most one pixel away.

#include <cmath>
#include <cstdio>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "core/common/synthetic.hpp"
#include "core/stereo/block_matching.hpp"
#include "core/stereo/seeds.hpp"

namespace {

using namespace vl::stereo;

struct Scene {
    vl::synth::StereoScene s;
    cv::Mat visible;
};
const Scene& scene() {
    static const Scene sc = [] {
        Scene x{vl::synth::layeredStereo({384, 288}, 32, 7), {}};
        cv::bitwise_not(x.s.occluded, x.visible);
        return x;
    }();
    return sc;
}

SeedParams defaults() {
    SeedParams p;
    p.maxDisparity = 36;
    p.seedSpacing = 16;
    p.stopScore = 0.5;
    return p;
}

TEST(SeedPropagation, ReachesMostOfATexturedSceneWithFewErrors) {
    const auto& sc = scene();
    const auto res = propagateSeeds(sc.s.left, sc.s.right, defaults());
    const cv::Mat reached = res.order >= 0;
    const double share = double(cv::countNonZero(reached)) / double(reached.total());
    const double bad = badPixelRate(res.disparity, sc.s.disparity, sc.visible & reached, 1.0);
    std::printf("seeds %zu, reached %.1f %%, bad among reached %.2f %%\n", res.seeds.size(), 100 * share, 100 * bad);
    EXPECT_GT(share, 0.9);
    EXPECT_LT(bad, 0.04);
    EXPECT_LT(res.seeds.size(), 500u);  // one per 16x16 cell at most
    EXPECT_EQ(int(cv::countNonZero(reached)), int(res.seeds.size()) + res.grown);
}

TEST(SeedPropagation, GrowsBestFirstFromNeighboursWithinOnePixelOfDisparity) {
    const auto& sc = scene();
    const auto p = defaults();
    const auto res = propagateSeeds(sc.s.left, sc.s.right, p);
    const int W = res.order.cols, H = res.order.rows;
    std::vector<int> seen(size_t(res.grown) + 1, 0);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const int o = res.order.at<int>(y, x);
            if (o < 0) {
                ASSERT_TRUE(std::isnan(res.disparity.at<float>(y, x)));
                continue;
            }
            ASSERT_GE(res.score.at<float>(y, x), o == 0 ? p.seedScore : p.stopScore);
            if (o == 0) continue;
            ++seen[size_t(o)];
            // Its parent: a 4-neighbour reached earlier, with a disparity at most 1 away.
            bool parent = false;
            const int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
            for (int k = 0; k < 4; ++k) {
                const int u = x + dx[k], v = y + dy[k];
                if (u < 0 || v < 0 || u >= W || v >= H) continue;
                const int ou = res.order.at<int>(v, u);
                if (ou >= 0 && ou < o && std::abs(res.disparity.at<float>(v, u) - res.disparity.at<float>(y, x)) <= 1.f)
                    parent = true;
            }
            ASSERT_TRUE(parent) << "pixel " << x << "," << y;
        }
    for (int o = 1; o <= res.grown; ++o) ASSERT_EQ(seen[size_t(o)], 1) << "order " << o;  // a permutation
}

TEST(SeedPropagation, AHigherFloorStopsEarlierButIsMoreAccurate) {
    const auto& sc = scene();
    auto p = defaults();
    p.stopScore = 0.3;
    const auto loose = propagateSeeds(sc.s.left, sc.s.right, p);
    p.stopScore = 0.9;
    const auto strict = propagateSeeds(sc.s.left, sc.s.right, p);
    EXPECT_LT(strict.grown, loose.grown);
    const double badLoose = badPixelRate(loose.disparity, sc.s.disparity, sc.visible & (loose.order >= 0), 1.0);
    const double badStrict = badPixelRate(strict.disparity, sc.s.disparity, sc.visible & (strict.order >= 0), 1.0);
    EXPECT_LE(badStrict, badLoose);
}

TEST(SeedPropagation, ObserverSeesTheRegionGrow) {
    const auto& sc = scene();
    int calls = 0, last = 0;
    const auto res = propagateSeeds(sc.s.left, sc.s.right, defaults(), {}, {},
                                    [&](const cv::Mat& d, const cv::Mat& order, int reached) {
                                        ++calls;
                                        EXPECT_GT(reached, last);
                                        EXPECT_EQ(d.size(), order.size());
                                        last = reached;
                                    }, 5000);
    EXPECT_EQ(calls, res.grown / 5000);
}

}  // namespace
