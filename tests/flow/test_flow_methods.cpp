// The estimators against exact synthetic flow, their key properties, and OpenCV's DIS as a reference.

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "core/common/cv_compat.hpp"
#include "core/common/image.hpp"
#include "core/common/synthetic.hpp"
#include "core/flow/block_flow.hpp"
#include "core/flow/flow_eval.hpp"
#include "core/flow/flow_scenes.hpp"
#include "core/flow/horn_schunck.hpp"
#include "core/flow/lucas_kanade.hpp"

using namespace vl::flow;

namespace {

const cv::Size kSize(256, 192);

// Pure translation; an integer shift makes frame 1 an exact copy of frame 0, shifted.
vl::synth::FlowPair translation(double dx, double dy) { return vl::synth::rigidFlow(kSize, 0.0, 1.0, {dx, dy}, 11); }

// Truth with a margin marked unknown: pixels that leave the frame or see the replicated border.
cv::Mat inner(const cv::Mat& truth, int margin) {
    cv::Mat t(truth.size(), CV_32FC2, cv::Scalar(1e10, 1e10));
    const cv::Rect r(margin, margin, truth.cols - 2 * margin, truth.rows - 2 * margin);
    truth(r).copyTo(t(r));
    return t;
}

double epe(const cv::Mat& flow, const cv::Mat& truth, int margin = 12) { return evaluateFlow(flow, inner(truth, margin)).epe; }

LucasKanadeParams lkParams(int levels) {
    LucasKanadeParams p;
    p.levels = levels;
    return p;
}

HornSchunckParams hsParams(int levels) {
    HornSchunckParams p;
    p.levels = levels;
    return p;
}

}  // namespace

// ------------------------------------------------------------------ Lucas–Kanade

TEST(LucasKanade, RecoversASmallTranslation) {
    const auto s = translation(1, -1);
    EXPECT_LT(epe(lucasKanade(s.frame0, s.frame1, lkParams(1)).flow, s.flow), 0.05);
    const auto t = translation(0.6, -0.4);  // sub-pixel: frame 1 is resampled, a little blur
    EXPECT_LT(epe(lucasKanade(t.frame0, t.frame1, lkParams(1)).flow, t.flow), 0.2);
}

TEST(LucasKanade, CoarseToFineFollowsALargeMotionThatOneScaleCannot) {
    const auto s = translation(7, -5);
    const double single = epe(lucasKanade(s.frame0, s.frame1, lkParams(1)).flow, s.flow);
    const double pyramid = epe(lucasKanade(s.frame0, s.frame1, lkParams(4)).flow, s.flow);
    EXPECT_GT(single, 2.0);
    EXPECT_LT(pyramid, 0.1);
}

TEST(LucasKanade, IterationsConvergeAndDoNotDrift) {
    const auto s = vl::synth::rigidFlow(kSize, 2.0, 1.02, {1.0, -0.5}, 5);
    auto p = lkParams(3);
    p.iterations = 3;
    const auto a = lucasKanade(s.frame0, s.frame1, p);
    p.iterations = 10;
    const auto b = lucasKanade(s.frame0, s.frame1, p);
    EXPECT_LT(b.meanUpdate.back(), 0.02);                   // a fixed point is reached
    EXPECT_LT(epe(b.flow, s.flow), epe(a.flow, s.flow) + 0.02);  // and more iterations do not hurt
}

TEST(LucasKanade, StructureTensorShowsTheApertureProblem) {
    // A flat square (x 60..140, y 50..130) on texture: an edge window has λ₂ << λ₁, a flat window has
    // both ~0, a corner pins both.
    const auto s = slidingSquare(kSize, {60, 50, 80, 80}, {2, 1}, 3);
    auto p = lkParams(1);
    p.sigma = 2.0;
    const auto r = lucasKanade(s.frame0, s.frame1, p);
    const auto l1 = [&](int x, int y) { return r.maxEigen.at<float>(y, x); };
    const auto l2 = [&](int x, int y) { return r.minEigen.at<float>(y, x); };
    EXPECT_LT(l2(100, 90), 1e-3f);                  // inside: nothing to see
    EXPECT_LT(l1(100, 90), 1e-3f);
    EXPECT_GT(l1(60, 90), 20.f * l2(60, 90));     // left edge: one strong direction only
    EXPECT_GT(l1(100, 50), 20.f * l2(100, 50));   // top edge
    EXPECT_GT(l2(60, 50), 0.1f * l1(60, 50));     // corner: both directions
    // On an edge only the normal component is recovered; just inside the corner both are (on the
    // corner pixel itself the window straddles the motion boundary and averages two motions).
    const cv::Vec2f left = r.flow.at<cv::Vec2f>(90, 60), top = r.flow.at<cv::Vec2f>(50, 100), corner = r.flow.at<cv::Vec2f>(53, 63);
    EXPECT_NEAR(left[0], 2.f, 0.3f);
    EXPECT_GT(std::abs(left[1] - 1.f), 0.5f);
    EXPECT_NEAR(top[1], 1.f, 0.3f);
    EXPECT_GT(std::abs(top[0] - 2.f), 0.5f);
    EXPECT_NEAR(corner[0], 2.f, 0.3f);
    EXPECT_NEAR(corner[1], 1.f, 0.3f);

    // Bouguet's reliability test removes the flat inside.
    p.minEigen = 5.0;
    const auto q = lucasKanade(s.frame0, s.frame1, p);
    EXPECT_TRUE(std::isnan(q.flow.at<cv::Vec2f>(90, 100)[0]));
    EXPECT_FALSE(std::isnan(q.flow.at<cv::Vec2f>(50, 60)[0]));
}

// ------------------------------------------------------------------ Horn–Schunck

TEST(HornSchunck, RecoversASmallTranslation) {
    // One scale: information spreads about a pixel per sweep, so give it enough sweeps.
    auto p = hsParams(1);
    p.iterations = 400;
    const auto s = translation(1, -1);
    EXPECT_LT(epe(hornSchunck(s.frame0, s.frame1, p).flow, s.flow), 0.05);
    const auto t = translation(0.6, -0.4);
    EXPECT_LT(epe(hornSchunck(t.frame0, t.frame1, p).flow, t.flow), 0.1);
}

TEST(HornSchunck, CoarseToFineFollowsALargeMotionThatOneScaleCannot) {
    const auto s = translation(7, -5);
    const double single = epe(hornSchunck(s.frame0, s.frame1, hsParams(1)).flow, s.flow);
    const double pyramid = epe(hornSchunck(s.frame0, s.frame1, hsParams(4)).flow, s.flow);
    EXPECT_GT(single, 2.0);
    EXPECT_LT(pyramid, 0.15);
}

TEST(HornSchunck, GaussSeidelNeverIncreasesTheEnergy) {
    // One linearisation (one level, one warp): the energy is a fixed quadratic and each Gauss–Seidel
    // update minimises it exactly in (u, v) at one pixel.
    const auto s = vl::synth::rigidFlow(kSize, 1.0, 1.01, {0.5, 0.3}, 2);
    auto p = hsParams(1);
    p.warps = 1;
    p.iterations = 60;
    p.solver = HornSchunckSolver::GaussSeidel;
    const auto r = hornSchunck(s.frame0, s.frame1, p);
    ASSERT_EQ(r.energy.size(), 61u);
    for (size_t i = 1; i < r.energy.size(); ++i) ASSERT_LE(r.energy[i], r.energy[i - 1] * (1 + 1e-6)) << "sweep " << i;
    EXPECT_LT(r.energy.back(), 0.5 * r.energy.front());
}

TEST(HornSchunck, JacobiAndGaussSeidelReachTheSameMinimiser) {
    cv::Mat I0(40, 50, CV_32F), Ix, Iy, It;
    cv::randu(I0, 0, 255);
    cv::GaussianBlur(I0, I0, cv::Size(), 2.0);
    cv::Sobel(I0, Ix, CV_32F, 1, 0, 3, 1.0 / 8);
    cv::Sobel(I0, Iy, CV_32F, 0, 1, 3, 1.0 / 8);
    It = -0.8f * Ix + 0.3f * Iy;  // brightness change of a (0.8, -0.3) translation
    cv::Mat u1 = cv::Mat::zeros(I0.size(), CV_32F), v1 = u1.clone(), u2 = u1.clone(), v2 = u1.clone();
    for (int i = 0; i < 4000; ++i) {
        hornSchunckSweep(u1, v1, Ix, Iy, It, 2.0, HornSchunckSolver::Jacobi);
        hornSchunckSweep(u2, v2, Ix, Iy, It, 2.0, HornSchunckSolver::GaussSeidel);
    }
    EXPECT_LT(cv::norm(u1, u2, cv::NORM_INF), 1e-3);
    EXPECT_LT(cv::norm(v1, v2, cv::NORM_INF), 1e-3);
    EXPECT_NEAR(cv::mean(u2)[0], 0.8, 0.05);
    EXPECT_NEAR(cv::mean(v2)[0], -0.3, 0.05);
}

TEST(HornSchunck, FillsInWhereLucasKanadeCannot) {
    // Inside the flat square (truth (2, 1)) there is no gradient: single-scale LK leaves it at zero,
    // HS propagates the motion of the edges inwards.
    const auto s = slidingSquare(kSize, {70, 50, 60, 60}, {2, 1}, 3);
    const auto lk = lucasKanade(s.frame0, s.frame1, lkParams(1));
    const auto hs = hornSchunck(s.frame0, s.frame1, hsParams(3));
    const cv::Rect centre(95, 75, 10, 10);
    const cv::Scalar lkMean = cv::mean(lk.flow(centre)), hsMean = cv::mean(hs.flow(centre));
    EXPECT_LT(std::hypot(lkMean[0], lkMean[1]), 0.1);
    EXPECT_GT(hsMean[0], 1.0);
    EXPECT_GT(hsMean[1], 0.4);
}

TEST(HornSchunck, ReportsPartialResultsAtFullResolution) {
    const auto s = translation(1, 0);
    auto p = hsParams(3);
    p.iterations = 20;
    p.reportEvery = 10;
    std::vector<std::string> stages;
    const auto r = hornSchunck(s.frame0, s.frame1, p, {}, {}, [&](const cv::Mat& f, const std::string& stage) {
        EXPECT_EQ(f.size(), kSize);
        stages.push_back(stage);
    });
    EXPECT_EQ(stages.size(), 6u);  // 2 per level
    EXPECT_EQ(r.sweeps, 60);
}

// ------------------------------------------------------------------ block matching

TEST(BlockMatching, IntegerTranslationIsExact) {
    const auto s = translation(3, -2);
    BlockFlowParams p;
    p.search = 5;
    p.subpixel = false;
    const auto r = blockMatchFlow(s.frame0, s.frame1, p);
    EXPECT_LT(epe(r.flow, s.flow), 1e-6);
    EXPECT_EQ(r.evaluations, long(kSize.area()) * 121);
    // The cost surface of one pixel has its minimum at the true displacement.
    const cv::Mat surf = blockCostSurface(vl::toGrayF(s.frame0), vl::toGrayF(s.frame1), {100, 80}, p);
    cv::Point best;
    cv::minMaxLoc(surf, nullptr, nullptr, &best);
    EXPECT_EQ(best, cv::Point(3 + 5, -2 + 5));
}

TEST(BlockMatching, SubPixelRefinementHelps) {
    // Parabola fits are biased towards whole pixels ("pixel locking"), but clearly beat rounding.
    const auto s = translation(2.3, 0.6);
    BlockFlowParams p;
    p.search = 4;
    p.cost = BlockCost::SSD;
    p.subpixel = false;
    const double integer = epe(blockMatchFlow(s.frame0, s.frame1, p).flow, s.flow);
    p.subpixel = true;
    const double sub = epe(blockMatchFlow(s.frame0, s.frame1, p).flow, s.flow);
    EXPECT_GT(integer, 0.55);
    EXPECT_LT(sub, 0.7 * integer);
}

TEST(BlockMatching, MotionBeyondTheSearchWindowIsNotFound) {
    const auto s = translation(6, 0);
    BlockFlowParams p;
    p.search = 3;
    EXPECT_GT(epe(blockMatchFlow(s.frame0, s.frame1, p).flow, s.flow), 2.0);
}

// ------------------------------------------------------------------ reference

TEST(FlowReference, SameBallparkAsOpenCvDis) {
    // The app's default sample: rotation, zoom and shift, up to 11 px of motion.
    const auto s = vl::synth::rigidFlow({384, 288}, 2.0, 1.02, {1.5, -1.0}, 7);
    cv::Mat g0, g1, dis;
    cv::cvtColor(s.frame0, g0, cv::COLOR_BGR2GRAY);
    cv::cvtColor(s.frame1, g1, cv::COLOR_BGR2GRAY);
    cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM)->calc(g0, g1, dis);
    const double ref = evaluateFlow(dis, s.flow).epe;
    const double lk = evaluateFlow(lucasKanade(s.frame0, s.frame1, {}).flow, s.flow).epe;
    const double hs = evaluateFlow(hornSchunck(s.frame0, s.frame1, {}).flow, s.flow).epe;
    EXPECT_LT(ref, 0.5);  // the reference itself works here
    EXPECT_LT(lk, 2 * ref + 0.05);
    EXPECT_LT(hs, 2 * ref + 0.05);
}
