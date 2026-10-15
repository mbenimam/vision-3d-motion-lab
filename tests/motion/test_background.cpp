#include <cmath>
#include <cstdio>
#include <random>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

#include "core/common/synthetic.hpp"
#include "core/motion/background.hpp"
#include "test_util.hpp"
#include "core/motion/regularize.hpp"

namespace {

using namespace vl::motion;

cv::Mat noisy(const cv::Mat& gray, double sigma, std::mt19937& rng) {
    std::normal_distribution<double> n(0, sigma);
    cv::Mat out(gray.size(), CV_8U);
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x)
            out.at<uchar>(y, x) = cv::saturate_cast<uchar>(gray.at<uchar>(y, x) + n(rng));
    return out;
}

enum class Reg { None, Morphology, Mrf };

// Mean F1 of a model over a sequence, the model starting from the temporal median.
double meanF1(const vl::synth::MovingScene& s, BgParams p, Reg reg = Reg::None) {
    const cv::Mat bg = medianBackground(s.frames, 15);
    BackgroundModel m(p, bg, noiseSigma(s.frames[0], bg));
    double sum = 0;
    for (size_t t = 0; t < s.frames.size(); ++t) {
        const BgStep st = m.apply(s.frames[t]);
        cv::Mat fg = st.foreground;
        if (reg == Reg::Morphology) fg = alternatingSequentialFilter(fg, 2);
        if (reg == Reg::Mrf) fg = icmPotts(detectionMargin(st.difference, st.threshold), fg, MrfParams{}).mask;
        sum += compareMasks(fg, s.masks[t]).f1();
    }
    return sum / double(s.frames.size());
}

TEST(SigmaDelta, ConvergesToAStaticBackgroundFromAWrongStart) {
    const cv::Mat scene = vl::synth::texture({96, 72}, 5);
    std::mt19937 rng(1);
    BgParams p;
    p.method = BgMethod::SigmaDelta;
    BackgroundModel m(p, cv::Mat(scene.size(), CV_8U, cv::Scalar(128)), 2.0);
    BgStep last;
    for (int t = 0; t < 400; ++t) last = m.apply(noisy(scene, 2.0, rng));
    // ±1 per frame: within 260 frames every pixel has reached its value; then it hovers around the median.
    cv::Mat err;
    cv::absdiff(m.background(), scene, err);
    double maxErr = 0;
    cv::minMaxLoc(err, nullptr, &maxErr);
    EXPECT_LE(maxErr, 6.0);
    EXPECT_LT(cv::mean(err)[0], 1.5);
    // V has learnt the noise: almost nothing is flagged on an empty scene.
    EXPECT_LT(cv::countNonZero(last.foreground) / double(scene.total()), 0.02);
}

TEST(SigmaDelta, DetectsTheMovingObject) {
    const auto s = vl::motion::test::noisyMovingObject({160, 120}, 120, 3);
    BgParams p;
    p.method = BgMethod::SigmaDelta;
    p.amplification = 3;
    const double f1 = meanF1(s, p);
    EXPECT_GT(f1, 0.97) << "mean F1 " << f1;
}

TEST(SigmaDelta, RegularisationRemovesTheNoiseOfALowThreshold) {
    const auto s = vl::motion::test::noisyMovingObject({160, 120}, 120, 3);
    BgParams p;
    p.method = BgMethod::SigmaDelta;
    p.amplification = 2;  // V ≈ 2 × the typical |I − M|: many noise pixels pass
    const double raw = meanF1(s, p), morph = meanF1(s, p, Reg::Morphology), mrf = meanF1(s, p, Reg::Mrf);
    std::printf("N = 2: mean F1 raw %.3f, morphology %.3f, MRF %.3f\n", raw, morph, mrf);
    EXPECT_LT(raw, 0.93);
    EXPECT_GT(morph, raw + 0.05);
    EXPECT_GT(mrf, raw + 0.05);
    EXPECT_GT(mrf, 0.97);
}

TEST(Background, EveryMethodDetectsTheObjectAtLeastAsWellAsOpenCvMog2) {
    const auto s = vl::motion::test::noisyMovingObject({160, 120}, 120, 3);
    // Reference: OpenCV's mixture of Gaussians on the same frames (shadows off, after a warm-up).
    auto mog = cv::createBackgroundSubtractorMOG2(60, 16, false);
    double ref = 0;
    int n = 0;
    for (int pass = 0; pass < 2; ++pass)
        for (size_t t = 0; t < s.frames.size(); ++t) {
            cv::Mat fg;
            mog->apply(s.frames[t], fg);
            if (pass == 1) {
                ref += compareMasks(fg, s.masks[t]).f1();
                ++n;
            }
        }
    ref /= n;
    for (BgMethod method : {BgMethod::Mean, BgMethod::Median, BgMethod::SigmaDelta, BgMethod::Gaussian}) {
        BgParams p;
        p.method = method;
        p.tau = 20;
        const double f1 = meanF1(s, p);
        std::printf("method %d: mean F1 %.3f (MOG2 %.3f)\n", int(method), f1, ref);
        EXPECT_GT(f1, 0.95) << "method " << int(method);
        EXPECT_GT(f1, ref) << "method " << int(method) << " vs MOG2 " << ref;
    }
}

TEST(Median, MovesOneGreyLevelPerFrameOnAStep) {
    BgParams p;
    p.method = BgMethod::Median;
    p.tau = 1000;  // never foreground: we only watch the estimate
    BackgroundModel m(p, cv::Mat(4, 4, CV_8U, cv::Scalar(50)));
    const cv::Mat step(4, 4, CV_8U, cv::Scalar(80));
    for (int t = 1; t <= 40; ++t) {
        m.apply(step);
        EXPECT_EQ(m.background().at<uchar>(2, 2), std::min(80, 50 + t)) << "frame " << t;
    }
}

TEST(Mean, DecaysExponentiallyOnAStep) {
    BgParams p;
    p.method = BgMethod::Mean;
    p.alpha = 0.1f;
    p.tau = 1000;
    BackgroundModel m(p, cv::Mat(4, 4, CV_8U, cv::Scalar(50)));
    const cv::Mat step(4, 4, CV_8U, cv::Scalar(80));
    for (int t = 0; t < 30; ++t) {
        // |I − M_{t}| before the update of frame t is 30 (1 − α)^t.
        const BgStep s = m.apply(step);
        EXPECT_NEAR(s.difference.at<float>(1, 1), 30.0 * std::pow(0.9, t), 1e-3) << "frame " << t;
    }
}

TEST(Gaussian, LearnsTheNoiseVariance) {
    const cv::Mat scene(64, 64, CV_8U, cv::Scalar(100));
    std::mt19937 rng(2);
    BgParams p;
    p.method = BgMethod::Gaussian;
    p.alpha = 0.02f;
    p.amplification = 3;  // threshold = 3σ
    BackgroundModel m(p, scene, 10.0);  // start far too wide
    BgStep s;
    for (int t = 0; t < 600; ++t) s = m.apply(noisy(scene, 4.0, rng));
    EXPECT_NEAR(cv::mean(s.threshold)[0] / 3.0, 4.0, 0.4);
}

TEST(Mrf, IcmNeverRaisesTheEnergyAndRemovesIsolatedNoise) {
    const int W = 80, H = 60;
    cv::Mat margin = cv::Mat::zeros(H, W, CV_32F);
    margin(cv::Rect(20, 15, 25, 20)).setTo(3.0f);  // a real object
    cv::Mat truth = cv::Mat::zeros(H, W, CV_8U);
    truth(cv::Rect(20, 15, 25, 20)).setTo(255);
    std::mt19937 rng(4);
    std::uniform_int_distribution<int> ux(0, W - 1), uy(0, H - 1);
    std::uniform_real_distribution<float> strength(1.05f, 2.0f);
    // Isolated false alarms just above the threshold, away from the object (next to it, strong
    // evidence rightly joins the object), and holes inside it.
    const cv::Rect near(17, 12, 31, 26);
    for (int i = 0; i < 120; ++i) {
        const int x = ux(rng), y = uy(rng);
        if (!near.contains(cv::Point(x, y))) margin.at<float>(y, x) = strength(rng);
    }
    for (int i = 0; i < 25; ++i) {
        const int x = 22 + ux(rng) % 21, y = 17 + uy(rng) % 16;
        margin.at<float>(y, x) = 0.3f;
    }
    const cv::Mat init = margin > 1.0f;
    MrfParams p;
    p.beta = 0.6;
    const MrfResult r = icmPotts(margin, init, p);
    ASSERT_GE(r.energy.size(), 2u);
    for (size_t i = 1; i < r.energy.size(); ++i) EXPECT_LE(r.energy[i], r.energy[i - 1] + 1e-9) << "sweep " << i;
    EXPECT_NEAR(r.energy.back(), mrfEnergy(margin, r.mask, p), 1e-6);
    cv::Mat diff;
    cv::compare(r.mask, truth, diff, cv::CMP_NE);
    EXPECT_EQ(cv::countNonZero(diff), 0) << "ICM should give back exactly the object";
    EXPECT_GT(r.changes, 100);
    EXPECT_LT(compareMasks(init, truth).f1(), 0.9);

    // With β = 0 the data term alone decides: the result is the per-pixel test r > 1.
    MrfParams p0;
    p0.beta = 0;
    cv::compare(icmPotts(margin, init, p0).mask, init, diff, cv::CMP_NE);
    EXPECT_EQ(cv::countNonZero(diff), 0);
}

TEST(Morphology, FilterRemovesSpecksAndKeepsBlobs) {
    cv::Mat m = cv::Mat::zeros(60, 80, CV_8U);
    cv::circle(m, {40, 30}, 12, 255, cv::FILLED);
    m.at<uchar>(5, 5) = m.at<uchar>(50, 70) = 255;
    m.at<uchar>(30, 40) = 0;  // a hole
    const cv::Mat f = alternatingSequentialFilter(m, 2);
    EXPECT_EQ(f.at<uchar>(5, 5), 0);
    EXPECT_EQ(f.at<uchar>(50, 70), 0);
    EXPECT_EQ(f.at<uchar>(30, 40), 255);
    EXPECT_NEAR(cv::countNonZero(f), cv::countNonZero(m), 30);
}

TEST(Background, MedianBootstrapIsTheEmptyScene) {
    const auto s = vl::motion::test::noisyMovingObject({120, 90}, 60, 9);
    const cv::Mat bg = medianBackground(s.frames, 15);
    // Compare with a frame where the object is masked out: away from it, they agree up to noise and drift.
    cv::Mat d;
    cv::absdiff(toGray8(bg), toGray8(s.frames[0]), d);
    EXPECT_LT(cv::mean(d, s.masks[0] == 0)[0], 6.0);
    EXPECT_GT(cv::mean(d, s.masks[0])[0], 15.0);  // the object at frame 0 is not in the median
}

}  // namespace
