#include <gtest/gtest.h>

#include <cmath>

#include "core/common/cv_compat.hpp"
#include "core/geometry/homography.hpp"
#include "core/geometry/ransac.hpp"
#include "core/geometry/scenes.hpp"

using namespace vl::geometry;

namespace {

// A strong but plausible perspective map of a 800x600 image (like the Graffiti pairs).
const cv::Matx33d kH(0.76, -0.30, 225.0, 0.33, 1.01, -77.0, 3.5e-4, -1.4e-5, 1.0);
const cv::Size kSize(800, 600);

std::vector<cv::Point2f> toF(const Points& p) {
    std::vector<cv::Point2f> out;
    for (const auto& q : p) out.emplace_back(float(q.x), float(q.y));
    return out;
}

}  // namespace

TEST(Dlt, ExactOnFourPoints) {
    const Points src = {{10, 20}, {700, 40}, {650, 560}, {30, 500}};
    Points dst;
    for (const auto& p : src) dst.push_back(transform(kH, p));
    for (bool normalise : {true, false}) {
        const auto H = dltHomography(src, dst, normalise);
        ASSERT_TRUE(H);
        for (size_t i = 0; i < src.size(); ++i) EXPECT_NEAR(cv::norm(transform(*H, src[i]) - dst[i]), 0, 1e-6);
        EXPECT_LT(cornerError(*H, kH, kSize), 1e-6);
    }
}

TEST(Dlt, RejectsCollinearSamples) {
    const Points src = {{0, 0}, {100, 100}, {200, 200}, {50, 300}};
    EXPECT_TRUE(collinearTriple(src.data()));
    const Points ok = {{0, 0}, {100, 0}, {100, 100}, {0, 100}};
    EXPECT_FALSE(collinearTriple(ok.data()));
}

TEST(Dlt, NormalisationReducesErrorOnPixelCoordinates) {
    // Same noisy inliers, two least-squares fits: Hartley normalisation must give the more accurate H.
    double errN = 0, errU = 0;
    for (std::uint32_t seed = 1; seed <= 10; ++seed) {
        const auto pp = scenes::homographyPairs(kH, kSize, 60, 0, 1.0, seed);
        errN += cornerError(*dltHomography(pp.a, pp.b, true), kH, kSize);
        errU += cornerError(*dltHomography(pp.a, pp.b, false), kH, kSize);
    }
    EXPECT_LT(errN, errU);
    EXPECT_LT(errN / 10, 3.0);
}

TEST(Ransac, AdaptiveTrialCount) {
    // N = log(1-p) / log(1-w^s): w = 0.5, s = 4, p = 0.99 -> 72 (textbook value).
    EXPECT_EQ(adaptiveTrials(0.5, 4, 0.99, 100000), 72);
    EXPECT_EQ(adaptiveTrials(0.5, 8, 0.99, 100000), 1177);
    EXPECT_EQ(adaptiveTrials(1.0, 4, 0.99, 1000), 1);
    EXPECT_EQ(adaptiveTrials(0.0, 4, 0.99, 1000), 1000);
    // More outliers, bigger sample or more confidence: never fewer trials.
    EXPECT_GT(adaptiveTrials(0.3, 4, 0.99, 1 << 30), adaptiveTrials(0.6, 4, 0.99, 1 << 30));
    EXPECT_GT(adaptiveTrials(0.6, 8, 0.99, 1 << 30), adaptiveTrials(0.6, 4, 0.99, 1 << 30));
    EXPECT_GT(adaptiveTrials(0.6, 4, 0.999, 1 << 30), adaptiveTrials(0.6, 4, 0.99, 1 << 30));
}

TEST(Ransac, StopsEarlyAndCurveIsMonotone) {
    const auto pp = scenes::homographyPairs(kH, kSize, 300, 200, 0.5, 3);  // 40 % outliers
    RansacParams p;
    p.threshold = 3;
    const auto e = estimateHomography(pp.a, pp.b, p);
    ASSERT_TRUE(e.ok);
    // With w = 0.6 and s = 4 the adaptive bound is ~34 trials: far below the cap.
    EXPECT_LT(e.report.trials, 200);
    EXPECT_EQ(int(e.report.bestInliersPerTrial.size()), e.report.trials);
    for (size_t i = 1; i < e.report.bestInliersPerTrial.size(); ++i)
        EXPECT_GE(e.report.bestInliersPerTrial[i], e.report.bestInliersPerTrial[i - 1]);
    // Reproducible: same seed, same trials.
    const auto e2 = estimateHomography(pp.a, pp.b, p);
    EXPECT_EQ(e2.report.trials, e.report.trials);
    EXPECT_EQ(e2.nInliers, e.nInliers);
}

TEST(Homography, RecoversTruthWithFortyPercentOutliers) {
    const auto pp = scenes::homographyPairs(kH, kSize, 300, 200, 0.7, 11);
    RansacParams p;
    p.threshold = 3;
    const auto e = estimateHomography(pp.a, pp.b, p);
    ASSERT_TRUE(e.ok);
    // Classification: (almost) every true inlier found, no outlier accepted by chance more than a few.
    int tp = 0, fp = 0;
    for (size_t i = 0; i < pp.a.size(); ++i) {
        tp += e.inliers[i] && pp.isInlier[i];
        fp += e.inliers[i] && !pp.isInlier[i];
    }
    EXPECT_GE(tp, 290);
    EXPECT_LE(fp, 3);
    const double ours = cornerError(e.H, kH, kSize);
    EXPECT_LT(ours, 1.0);  // pixels, with 0.7 px noise on both images
    EXPECT_LT(e.rms, 1.5);

    // OpenCV's RANSAC on the same data, as a reference: we must be as accurate (within a margin).
    const cv::Mat Hcv = cv::findHomography(toF(pp.a), toF(pp.b), cv::RANSAC, 3.0);
    ASSERT_FALSE(Hcv.empty());
    const double ref = cornerError(cv::Matx33d(Hcv), kH, kSize);
    EXPECT_LT(ours, ref + 0.5);
}
