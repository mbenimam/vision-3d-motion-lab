#include <gtest/gtest.h>

#include <cmath>

#include "core/common/cv_compat.hpp"
#include "core/geometry/fundamental.hpp"
#include "core/geometry/scenes.hpp"

using namespace vl::geometry;

namespace {

const cv::Size kSize(1024, 768);
const cv::Matx33d kK(800, 0, 512, 0, 800, 384, 0, 0, 1);
const cv::Matx33d kR = scenes::rotation(-8, 3, 2);
const cv::Vec3d kC2(0.6, -0.1, 1.0);  // camera 2 centre, in camera 1's frame
const cv::Vec3d kT = -(kR * kC2);

double meanTrueResidual(const cv::Matx33d& F, const scenes::PointPairs& pp) {
    return meanSymmetricEpipolarDistance(F, pp.cleanA, pp.cleanB);
}

}  // namespace

TEST(Fundamental, TrueFSatisfiesEpipolarConstraint) {
    const auto pp = scenes::twoViewPairs(kK, kR, kT, kSize, 100, 0, 0.0, 1);
    const cv::Matx33d F = fundamentalFromCameras(kK, kK, kR, kT);
    for (size_t i = 0; i < pp.a.size(); ++i) {
        EXPECT_NEAR(cv::Vec3d(pp.b[i].x, pp.b[i].y, 1).dot(F * cv::Vec3d(pp.a[i].x, pp.a[i].y, 1)), 0, 1e-9);
        EXPECT_LT(sampsonDistance(F, pp.a[i], pp.b[i]), 1e-6);
    }
}

TEST(Fundamental, EightPointIsRankTwoAndFitsTruePairs) {
    const auto pp = scenes::twoViewPairs(kK, kR, kT, kSize, 150, 0, 0.5, 2);
    const auto F = eightPoint(pp.a, pp.b);
    ASSERT_TRUE(F);
    cv::Matx31d s;
    cv::SVD::compute(*F, s);
    EXPECT_LT(s(2) / s(1), 1e-10);       // rank 2 enforced
    EXPECT_LT(meanTrueResidual(*F, pp), 0.5);  // pixels, noise-free pairs against the estimated lines

    // Without the projection the noisy least-squares solution is full rank.
    const auto F3 = eightPoint(pp.a, pp.b, {true, false});
    cv::SVD::compute(*F3, s);
    EXPECT_GT(s(2) / s(1), 1e-6);
}

TEST(Fundamental, EpipoleIsTheOtherCameraCentre) {
    const auto pp = scenes::twoViewPairs(kK, kR, kT, kSize, 200, 0, 0.3, 3);
    const auto F = eightPoint(pp.a, pp.b);
    ASSERT_TRUE(F);
    const auto e = epipoles(*F);
    // e1 = image of camera 2's centre in camera 1, e2 = image of camera 1's centre (origin) in camera 2.
    const cv::Vec3d trueE1 = kK * kC2, trueE2 = kK * kT;
    const auto p1 = finitePoint(e.e1), t1 = finitePoint(trueE1);
    const auto p2 = finitePoint(e.e2), t2 = finitePoint(trueE2);
    ASSERT_TRUE(p1 && t1 && p2 && t2);
    EXPECT_LT(cv::norm(*p1 - *t1), 10.0);  // pixels (the epipole is the least stable part of F)
    EXPECT_LT(cv::norm(*p2 - *t2), 10.0);
    EXPECT_LT(cv::norm(*F * e.e1), 1e-9);
    EXPECT_LT(cv::norm(F->t() * e.e2), 1e-9);
}

TEST(Fundamental, NormalisationMatters) {
    // Hartley's point: on pixel coordinates the unnormalised 8-point is clearly less accurate. A mostly
    // sideways baseline (epipole far outside the image) makes the effect large: ~4x here.
    const cv::Matx33d R = scenes::rotation(-8, 3, 2);
    const cv::Vec3d t = -(R * cv::Vec3d(1.0, 0.0, 0.1));
    double errN = 0, errU = 0;
    for (std::uint32_t seed = 1; seed <= 20; ++seed) {
        const auto pp = scenes::twoViewPairs(kK, R, t, kSize, 60, 0, 1.0, seed);
        errN += meanTrueResidual(*eightPoint(pp.a, pp.b, {true, true}), pp);
        errU += meanTrueResidual(*eightPoint(pp.a, pp.b, {false, true}), pp);
    }
    EXPECT_LT(2.0 * errN, errU);
}

TEST(Fundamental, EightPointMatchesOpenCv) {
    const auto pp = scenes::twoViewPairs(kK, kR, kT, kSize, 80, 0, 1.0, 9);
    const auto F = eightPoint(pp.a, pp.b);
    std::vector<cv::Point2f> a, b;
    for (size_t i = 0; i < pp.a.size(); ++i) a.emplace_back(pp.a[i]), b.emplace_back(pp.b[i]);
    cv::Matx33d ref(cv::findFundamentalMat(a, b, cv::FM_8POINT));
    ref *= 1.0 / cv::norm(ref);
    if (ref.dot(*F) < 0) ref *= -1.0;  // same matrix up to sign
    EXPECT_LT(cv::norm(*F - ref), 1e-4);
}

TEST(Fundamental, RansacWithOutliersMatchesOpenCv) {
    const auto pp = scenes::twoViewPairs(kK, kR, kT, kSize, 300, 200, 0.5, 4);  // 40 % outliers
    RansacParams p;
    p.threshold = 1.5;
    const auto e = estimateFundamental(pp.a, pp.b, p);
    ASSERT_TRUE(e.ok);
    int tp = 0, fp = 0;
    for (size_t i = 0; i < pp.a.size(); ++i) {
        tp += e.inliers[i] && pp.isInlier[i];
        fp += e.inliers[i] && !pp.isInlier[i];
    }
    EXPECT_GE(tp, 280);
    EXPECT_LE(fp, 15);  // an outlier can fall near its epipolar line by chance: a 1D constraint
    EXPECT_LT(e.singularValues[2], 1e-10);
    const double ours = meanTrueResidual(e.F, pp);
    EXPECT_LT(ours, 0.5);

    std::vector<cv::Point2f> a, b;
    for (size_t i = 0; i < pp.a.size(); ++i) a.emplace_back(pp.a[i]), b.emplace_back(pp.b[i]);
    const cv::Mat Fcv = cv::findFundamentalMat(a, b, cv::FM_RANSAC, 1.5, 0.99);
    ASSERT_FALSE(Fcv.empty());
    EXPECT_LT(ours, meanTrueResidual(cv::Matx33d(Fcv(cv::Rect(0, 0, 3, 3))), pp) + 0.3);
}

TEST(Fundamental, ClipLineToImage) {
    // Horizontal line y = 100 across a 200 x 150 image.
    const auto seg = clipLine({0, 1, -100}, {200, 150});
    ASSERT_TRUE(seg);
    EXPECT_NEAR(seg->first.y, 100, 1e-9);
    EXPECT_NEAR(std::abs(seg->second.x - seg->first.x), 200, 1e-9);
    EXPECT_FALSE(clipLine({0, 1, -400}, {200, 150}));
}

TEST(Fundamental, SyntheticRoomHasConsistentGroundTruth) {
    const auto s = scenes::room({320, 240}, 5);
    ASSERT_GT(s.x1.size(), 200u);
    for (size_t i = 0; i < s.x1.size(); ++i) EXPECT_LT(sampsonDistance(s.F, s.x1[i], s.x2[i]), 1e-6);
}
