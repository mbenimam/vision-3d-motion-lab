// Kolmogorov–Zabih graph-cut stereo: each expansion move is the exact optimum of its move space
// (checked by enumeration on tiny images), the energy never rises, and on the synthetic layered
// scene the result beats block matching and finds the true occlusions.

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "core/common/synthetic.hpp"
#include "core/stereo/block_matching.hpp"
#include "core/stereo/graphcut_stereo.hpp"

namespace {

using namespace vl::stereo;
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

cv::Mat randomImage(std::mt19937& rng, int W, int H) {
    cv::Mat img(H, W, CV_8UC3);
    std::uniform_int_distribution<int> v(0, 255), coin(0, 1);
    // Flat runs and jumps, so that both smoothness weights (λ and λ/3) occur.
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < 3; ++c)
                img.at<cv::Vec3b>(y, x)[c] = (x > 0 && coin(rng)) ? img.at<cv::Vec3b>(y, x - 1)[c] : uchar(v(rng));
    return img;
}

// A random unique configuration: each left pixel occluded or at a disparity whose right pixel is free.
cv::Mat randomUnique(std::mt19937& rng, int W, int H, int maxD) {
    cv::Mat d(H, W, CV_32F, cv::Scalar(kNaN));
    std::uniform_int_distribution<int> pick(-1, maxD);
    for (int y = 0; y < H; ++y) {
        std::vector<bool> taken(size_t(W), false);
        for (int x = 0; x < W; ++x) {
            const int k = pick(rng);
            if (k < 0 || x - k < 0 || taken[size_t(x - k)]) continue;
            taken[size_t(x - k)] = true;
            d.at<float>(y, x) = float(k);
        }
    }
    return d;
}

// Minimum energy over every configuration within one α-expansion of `init`, by enumeration: each left
// pixel keeps its current assignment (if any, and not α), takes α, or becomes occluded.
double bruteForceExpansion(const cv::Mat& L, const cv::Mat& R, const cv::Mat& init, int alpha, const GraphCutParams& p) {
    const int n = init.cols * init.rows;
    std::vector<std::vector<float>> options(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int x = i % init.cols;
        const float d = init.at<float>(i / init.cols, x);
        auto& o = options[size_t(i)];
        if (p.modelOcclusions) o.push_back(kNaN);
        if (std::isfinite(d) && int(d) != alpha) o.push_back(d);
        if (p.modelOcclusions ? x - alpha >= 0 : true) o.push_back(float(alpha));
    }
    double best = std::numeric_limits<double>::infinity();
    std::vector<size_t> pick(size_t(n), 0);
    cv::Mat cfg = init.clone();
    for (;;) {
        for (int i = 0; i < n; ++i) cfg.at<float>(i / init.cols, i % init.cols) = options[size_t(i)][pick[size_t(i)]];
        best = std::min(best, graphCutEnergy(L, R, cfg, p));  // +inf when not unique
        int k = 0;
        while (k < n && ++pick[size_t(k)] == options[size_t(k)].size()) pick[size_t(k++)] = 0;
        if (k == n) break;
    }
    return best;
}

TEST(GraphCutStereo, ExpansionMoveIsTheExactOptimumOfItsMoveSpace) {
    std::mt19937 rng(5);
    for (int trial = 0; trial < 12; ++trial) {
        const int W = 5, H = 2, maxD = 2;
        const cv::Mat L = randomImage(rng, W, H), R = randomImage(rng, W, H);
        GraphCutParams p;
        p.maxDisparity = maxD;
        p.cost = trial % 2 ? DataCost::Squared : DataCost::Absolute;
        p.birchfieldTomasi = trial % 3 != 0;
        p.lambda = 4 + trial;  // make smoothness, data and occlusion compete
        p.occlusionCost = 6 + 2 * (trial % 4);
        const cv::Mat init = randomUnique(rng, W, H, maxD);
        for (int alpha = 0; alpha <= maxD; ++alpha) {
            GraphCutStereo s(L, R, p);
            ASSERT_TRUE(s.setDisparity(init));
            const double before = s.energy();
            s.expand(alpha);
            const double expected = bruteForceExpansion(L, R, init, alpha, p);
            ASSERT_NEAR(s.energy(), expected, 1e-9) << "trial " << trial << ", alpha " << alpha;
            ASSERT_LE(s.energy(), before);
        }
    }
}

TEST(GraphCutStereo, PottsExpansionMoveIsTheExactOptimumOfItsMoveSpace) {
    std::mt19937 rng(9);
    for (int trial = 0; trial < 8; ++trial) {
        const int W = 6, H = 2, maxD = 3;
        const cv::Mat L = randomImage(rng, W, H), R = randomImage(rng, W, H);
        GraphCutParams p;
        p.maxDisparity = maxD;
        p.modelOcclusions = false;
        p.lambda = 5 + 3 * trial;
        cv::Mat init(H, W, CV_32F);
        std::uniform_int_distribution<int> lab(0, maxD);
        for (int i = 0; i < W * H; ++i) init.at<float>(i / W, i % W) = float(lab(rng));
        for (int alpha = 0; alpha <= maxD; ++alpha) {
            GraphCutStereo s(L, R, p);
            ASSERT_TRUE(s.setDisparity(init));
            s.expand(alpha);
            ASSERT_NEAR(s.energy(), bruteForceExpansion(L, R, init, alpha, p), 1e-9) << "trial " << trial;
        }
    }
}

TEST(GraphCutStereo, SetDisparityRejectsTwoLeftPixelsOnOneRightPixel) {
    std::mt19937 rng(2);
    const cv::Mat L = randomImage(rng, 6, 1), R = randomImage(rng, 6, 1);
    GraphCutParams p;
    p.maxDisparity = 3;
    cv::Mat d(1, 6, CV_32F, cv::Scalar(kNaN));
    d.at<float>(0, 3) = 1;  // right pixel 2
    d.at<float>(0, 4) = 2;  // right pixel 2 again
    EXPECT_TRUE(std::isinf(graphCutEnergy(L, R, d, p)));
    d.at<float>(0, 4) = 1;
    EXPECT_TRUE(std::isfinite(graphCutEnergy(L, R, d, p)));
}

struct Scene {
    vl::synth::StereoScene s;
    cv::Mat visible;
};
const Scene& smallScene() {
    static const Scene sc = [] {
        Scene x{vl::synth::layeredStereo({192, 144}, 16, 7), {}};
        cv::bitwise_not(x.s.occluded, x.visible);
        return x;
    }();
    return sc;
}

GraphCutParams sceneParams() {
    GraphCutParams p;
    p.maxDisparity = 18;
    p.cost = DataCost::Absolute;
    p.lambda = 2;
    p.occlusionCost = 6;
    p.maxIterations = 3;
    return p;
}

TEST(GraphCutStereo, EnergyNeverIncreasesAndMatchesTheReturnedMap) {
    const auto& sc = smallScene();
    const auto p = sceneParams();
    int observed = 0;
    const auto res = graphCutStereo(sc.s.left, sc.s.right, p, {}, {}, [&](const GraphCutStep& st) {
        ++observed;
        ASSERT_NE(st.disparity, nullptr);
    });
    ASSERT_EQ(int(res.energy.size()), res.moves + 1);
    EXPECT_EQ(observed, res.moves);
    for (size_t k = 1; k < res.energy.size(); ++k) ASSERT_LE(res.energy[k], res.energy[k - 1]) << "move " << k;
    EXPECT_LT(res.energy.back(), 0.2 * res.energy.front());
    EXPECT_NEAR(graphCutEnergy(sc.s.left, sc.s.right, res.disparity, p), res.energy.back(), 1e-6);
    EXPECT_NEAR(res.dataEnergy + res.occlusionEnergy + res.smoothEnergy, res.energy.back(), 1e-6);
}

TEST(GraphCutStereo, BeatsBlockMatchingOnTheLayeredScene) {
    const auto& sc = smallScene();
    BlockMatchParams bp;
    bp.maxDisparity = 18;
    const double bm = badPixelRate(blockMatch(sc.s.left, sc.s.right, bp).disparity, sc.s.disparity, sc.visible, 1.0);
    const auto gc = graphCutStereo(sc.s.left, sc.s.right, sceneParams());
    const double g = badPixelRate(gc.disparity, sc.s.disparity, sc.visible, 1.0);
    std::printf("bad pixels (non-occluded): block matching %.2f %%, graph cuts %.2f %%\n", 100 * bm, 100 * g);
    EXPECT_LT(g, 0.025);
    EXPECT_LT(g, 0.4 * bm);
}

TEST(GraphCutStereo, OcclusionsOverlapTheTrueOnes) {
    const auto& sc = smallScene();
    const auto gc = graphCutStereo(sc.s.left, sc.s.right, sceneParams());
    const int truth = cv::countNonZero(sc.s.occluded), found = cv::countNonZero(gc.occluded);
    const int both = cv::countNonZero(sc.s.occluded & gc.occluded);
    const double recall = double(both) / truth, precision = double(both) / found;
    std::printf("occlusions: recall %.2f, precision %.2f (%d true, %d found)\n", recall, precision, truth, found);
    EXPECT_GT(recall, 0.75);
    EXPECT_GT(precision, 0.7);
}

TEST(GraphCutStereo, PottsModeLabelsEveryPixel) {
    const auto& sc = smallScene();
    auto p = sceneParams();
    p.modelOcclusions = false;
    p.maxIterations = 2;
    const auto gc = graphCutStereo(sc.s.left, sc.s.right, p);
    EXPECT_EQ(cv::countNonZero(gc.occluded), 0);
    EXPECT_EQ(gc.occlusionEnergy, 0);
    for (size_t k = 1; k < gc.energy.size(); ++k) ASSERT_LE(gc.energy[k], gc.energy[k - 1]);
}

TEST(OcclusionsFromDisparity, AgreesWithTheSyntheticGroundTruth) {
    const auto s = vl::synth::layeredStereo({384, 288}, 32, 7);
    const cv::Mat occ = occlusionsFromDisparity(s.disparity);
    const int inter = cv::countNonZero(occ & s.occluded), uni = cv::countNonZero(occ | s.occluded);
    EXPECT_GT(double(inter) / uni, 0.9);
}

}  // namespace
