// Our implementations against OpenCV's on the same input: where both compute the same quantity they
// must agree (to rounding, or to the pixel when OpenCV works on an integer grid); where the methods
// differ (background models, trackers) ours must do at least as well on the synthetic ground truth.

#include <cmath>
#include <cstdio>
#include <random>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "core/common/synthetic.hpp"
#include "core/motion/background.hpp"
#include "core/motion/hough.hpp"
#include "core/motion/kalman.hpp"
#include "core/motion/mean_shift.hpp"
#include "core/motion/opencv_baselines.hpp"
#include "core/motion/regularize.hpp"
#include "test_util.hpp"

namespace {

using namespace vl::motion;

TEST(VsOpenCv, KalmanAgreesWithCvKalmanFilter) {
    KalmanParams kp;
    kp.processNoise = 0.3;
    kp.measurementNoise = 1.5;
    ConstantVelocityKalman ours(kp);
    ours.init({40, 25});

    // The same model in cv::KalmanFilter (double precision): A, H, Q, R and the start Θ_0, P_0.
    cv::KalmanFilter cvkf(4, 2, 0, CV_64F);
    cvkf.transitionMatrix = cv::Mat(ours.A());
    cvkf.measurementMatrix = cv::Mat(ours.H());
    cvkf.processNoiseCov = cv::Mat(ours.Q());
    cvkf.measurementNoiseCov = cv::Mat(ours.R());
    cvkf.statePost = cv::Mat(ours.state());
    cvkf.errorCovPost = cv::Mat(ours.covariance());

    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0, 1.5);
    for (int t = 1; t <= 50; ++t) {
        ours.predict();
        const cv::Mat predicted = cvkf.predict();
        for (int i = 0; i < 4; ++i) EXPECT_NEAR(ours.state()[i], predicted.at<double>(i), 1e-9) << "predict, frame " << t;
        // Some frames without a measurement (an occlusion): only predict runs.
        if (t % 10 >= 7) {
            cvkf.statePost = cvkf.statePre.clone();
            cvkf.errorCovPost = cvkf.errorCovPre.clone();
            continue;
        }
        const cv::Point2d X(40 + 2.0 * t + noise(rng), 25 - 1.0 * t + noise(rng));
        ours.update(X);
        const cv::Mat corrected = cvkf.correct(cv::Mat(cv::Vec2d(X.x, X.y)));
        for (int i = 0; i < 4; ++i) EXPECT_NEAR(ours.state()[i], corrected.at<double>(i), 1e-9) << "update, frame " << t;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                EXPECT_NEAR(ours.covariance()(i, j), cvkf.errorCovPost.at<double>(i, j), 1e-9) << "P, frame " << t;
    }
}

TEST(VsOpenCv, HueHistogramAndBackProjectionMatchCalcHist) {
    const cv::Mat img = vl::synth::colorTexture({120, 90}, 3);
    HueParams hp;
    hp.bins = 30;
    const HueImage h = hueOf(img, hp);
    const cv::Rect box(30, 20, 50, 40);
    const std::vector<float> ours = hueHistogram(h, box, hp.bins);

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    const cv::Mat roi = hsv(box), mask = h.valid(box);
    const int channels[] = {0}, histSize[] = {hp.bins};
    const float hueRange[] = {0, 180};
    const float* ranges[] = {hueRange};
    cv::Mat hist;
    cv::calcHist(&roi, 1, channels, mask, hist, 1, histSize, ranges);
    const double total = cv::sum(hist)[0];
    ASSERT_GT(total, 100);
    for (int b = 0; b < hp.bins; ++b) EXPECT_NEAR(ours[size_t(b)], hist.at<float>(b) / total, 1e-6) << "bin " << b;

    // Back-projection of the histogram scaled to a peak of 255, on the valid pixels.
    double peak = 0;
    cv::minMaxLoc(hist, nullptr, &peak);
    cv::Mat scaled = hist * (255.0 / peak), theirs;
    cv::calcBackProject(&hsv, 1, channels, scaled, theirs, ranges);
    const cv::Mat R = backProject(h, ours);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            if (h.valid.at<uchar>(y, x)) {
                EXPECT_NEAR(255.0 * R.at<float>(y, x), theirs.at<uchar>(y, x), 1.0) << x << "," << y;
            }
}

TEST(VsOpenCv, MeanShiftReachesTheSameModeAsCvMeanShift) {
    cv::Mat w(120, 160, CV_32F);
    const cv::Point2d mode(91.5, 57.5);
    for (int y = 0; y < w.rows; ++y)
        for (int x = 0; x < w.cols; ++x) {
            const double dx = x + 0.5 - mode.x, dy = y + 0.5 - mode.y;
            w.at<float>(y, x) = float(std::exp(-(dx * dx + 0.6 * dy * dy) / (2 * 8.0 * 8.0)));
        }
    for (const cv::Point2d start : {cv::Point2d(75, 45), cv::Point2d(105, 70), cv::Point2d(80, 66)}) {
        const cv::Size2d size(31, 31);
        const MeanShiftResult r = meanShift(w, start, size, 50, 0.1);
        opencv::ShiftTracker cvms(opencv::Shift::MeanShift, cv::Rect2d(start.x - 15.5, start.y - 15.5, 31, 31), 50, 0.1);
        const auto s = cvms.step(w);
        const cv::Point2d theirs(s.box.x + 0.5 * s.box.width, s.box.y + 0.5 * s.box.height);
        // OpenCV moves an integer window, so it can stop up to a pixel short of the mode.
        EXPECT_NEAR(r.centre.x, mode.x, 0.3) << start;
        EXPECT_NEAR(r.centre.y, mode.y, 0.3) << start;
        EXPECT_NEAR(theirs.x, r.centre.x, 1.0) << start;
        EXPECT_NEAR(theirs.y, r.centre.y, 1.0) << start;
    }
}

TEST(VsOpenCv, MorphologyMatchesMorphologyExWithTheSameDisc) {
    std::mt19937 rng(5);
    std::bernoulli_distribution on(0.1);
    cv::Mat m = cv::Mat::zeros(60, 80, CV_8U);
    cv::circle(m, {40, 30}, 14, 255, cv::FILLED);
    for (int y = 0; y < m.rows; ++y)
        for (int x = 0; x < m.cols; ++x)
            if (on(rng)) m.at<uchar>(y, x) = 255 - m.at<uchar>(y, x);  // specks and holes
    cv::Mat theirs = m.clone();
    for (int r = 1; r <= 2; ++r) {
        cv::Mat disc = cv::Mat::zeros(2 * r + 1, 2 * r + 1, CV_8U);
        for (int j = -r; j <= r; ++j)
            for (int i = -r; i <= r; ++i)
                if (i * i + j * j <= r * r) disc.at<uchar>(j + r, i + r) = 1;
        cv::morphologyEx(theirs, theirs, cv::MORPH_OPEN, disc, {-1, -1}, 1, cv::BORDER_REPLICATE);
        cv::morphologyEx(theirs, theirs, cv::MORPH_CLOSE, disc, {-1, -1}, 1, cv::BORDER_REPLICATE);
    }
    cv::Mat diff;
    cv::compare(alternatingSequentialFilter(m, 2), theirs, diff, cv::CMP_NE);
    EXPECT_EQ(cv::countNonZero(diff), 0);
}

TEST(VsOpenCv, HoughFindsTheSameTranslationAsGeneralizedHoughBallard) {
    const cv::Mat scene = vl::synth::colorTexture({200, 150}, 21);
    const cv::Rect2d box(60, 40, 41, 35);
    const cv::Point shift(-13, 11);
    cv::Mat moved;
    cv::warpAffine(scene, moved, cv::Matx23d(1, 0, shift.x, 0, 1, shift.y), scene.size(), cv::INTER_NEAREST,
                   cv::BORDER_REFLECT);
    HoughTrackerParams p;
    HoughTracker ours(p, scene, box);
    const cv::Point2d found = centreOf(ours.step(moved).step.box);
    opencv::BallardTracker theirs(scene, box, opencv::BallardParams{});
    const auto s = theirs.step(moved);
    const cv::Point2d expected = centreOf(box) + cv::Point2d(shift);
    EXPECT_NEAR(found.x, expected.x, 1e-9);
    EXPECT_NEAR(found.y, expected.y, 1e-9);
    ASSERT_FALSE(s.lost);
    // Ballard's reference point is the template's integer centre: within a pixel of ours.
    EXPECT_NEAR(centreOf(s.box).x, expected.x, 1.0);
    EXPECT_NEAR(centreOf(s.box).y, expected.y, 1.0);
}

// Mean F1 of a detector on the synthetic clip.
template <typename Detect>
double meanF1(const vl::synth::MovingScene& s, Detect detect) {
    double sum = 0;
    for (size_t t = 0; t < s.frames.size(); ++t) sum += compareMasks(detect(s.frames[t]), s.masks[t]).f1();
    return sum / double(s.frames.size());
}

TEST(VsOpenCv, SigmaDeltaWithMrfBeatsMog2AndKnnFromTheSameStart) {
    const auto s = vl::motion::test::noisyMovingObject({160, 120}, 120, 3);
    const cv::Mat bg = medianBackground(s.frames, 15);
    BackgroundModel model(BgParams{}, bg, noiseSigma(s.frames[1], s.frames[0]) / std::sqrt(2.0));
    const double ours = meanF1(s, [&](const cv::Mat& f) {
        const BgStep st = model.apply(f);
        return icmPotts(detectionMargin(st.difference, st.threshold), st.foreground, MrfParams{}).mask;
    });
    opencv::BackgroundSubtractor mog(opencv::Subtractor::MOG2, bg), knn(opencv::Subtractor::KNN, bg);
    const double mog2 = meanF1(s, [&](const cv::Mat& f) { return mog.apply(f).foreground; });
    const double knnF1 = meanF1(s, [&](const cv::Mat& f) { return knn.apply(f).foreground; });
    std::printf("mean F1: ours (Σ-Δ + MRF) %.3f, MOG2 %.3f, KNN %.3f\n", ours, mog2, knnF1);
    EXPECT_GT(mog2, 0.5);  // the baseline works, it is not a strawman
    EXPECT_GT(knnF1, 0.5);
    EXPECT_GT(ours, mog2);
    EXPECT_GT(ours, knnF1);
}

}  // namespace
