#include <cmath>
#include <random>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "core/common/synthetic.hpp"
#include "core/motion/hough.hpp"
#include "core/motion/kalman.hpp"
#include "core/motion/mean_shift.hpp"
#include "core/motion/sequences.hpp"
#include "test_util.hpp"

namespace {

using namespace vl::motion;

TEST(MeanShift, ClimbsOntoABlobFromANearbyStart) {
    cv::Mat w(90, 120, CV_32F);
    const cv::Point2d c(61.5, 44.5);
    for (int y = 0; y < w.rows; ++y)
        for (int x = 0; x < w.cols; ++x) {
            const double dx = x + 0.5 - c.x, dy = y + 0.5 - c.y;
            w.at<float>(y, x) = float(std::exp(-(dx * dx + dy * dy) / (2 * 6.0 * 6.0)));
        }
    for (const cv::Point2d start : {cv::Point2d(70, 52), cv::Point2d(52, 38), cv::Point2d(66, 35)}) {
        const auto r = meanShift(w, start, {25, 25}, 50, 0.01);
        EXPECT_TRUE(r.converged);
        EXPECT_NEAR(r.centre.x, c.x, 0.2) << start;
        EXPECT_NEAR(r.centre.y, c.y, 0.2) << start;
        EXPECT_GT(r.iterations, 1);
    }
}

TEST(MeanShift, BackProjectionIsTheModelDensity) {
    cv::Mat img(40, 40, CV_8UC3, cv::Scalar(30, 90, 235));  // orange
    img(cv::Rect(0, 0, 20, 40)).setTo(cv::Scalar(235, 120, 30));  // blue half
    img(cv::Rect(0, 30, 40, 10)).setTo(cv::Scalar(128, 128, 128));  // grey: no hue
    HueParams hp;
    const HueImage h = hueOf(img, hp);
    const auto hist = hueHistogram(h, {20, 0, 20, 40}, 16);  // orange + grey strip
    float sum = 0;
    for (float f : hist) sum += f;
    EXPECT_NEAR(sum, 1.f, 1e-5);
    const cv::Mat bp = backProject(h, hist);
    EXPECT_FLOAT_EQ(bp.at<float>(10, 30), 1.f);  // orange
    EXPECT_FLOAT_EQ(bp.at<float>(10, 5), 0.f);   // blue
    EXPECT_FLOAT_EQ(bp.at<float>(35, 30), 0.f);  // grey is masked
}

double meanIou(const vl::synth::MovingScene& s, const std::vector<cv::Rect2d>& est) {
    double sum = 0;
    for (size_t t = 0; t < est.size(); ++t) sum += iou(est[t], cv::Rect2d(s.boxes[t]));
    return sum / double(est.size());
}

TEST(MeanShift, TracksTheSyntheticObject) {
    const auto s = vl::motion::test::noisyMovingObject({320, 240}, 120, 11);
    MeanShiftTrackerParams p;
    MeanShiftTracker tr(p, s.frames[0], s.boxes[0]);
    std::vector<cv::Rect2d> est{tr.first().step.box};
    int lost = 0;
    for (size_t t = 1; t < s.frames.size(); ++t) {
        const auto f = tr.step(s.frames[t]);
        est.push_back(f.step.box);
        lost += f.step.lost;
    }
    const double m = meanIou(s, est);
    EXPECT_GT(m, 0.75) << "mean IoU " << m;
    EXPECT_EQ(lost, 0);
}

TEST(Hough, FindsAnExactTranslation) {
    const cv::Mat scene = vl::synth::colorTexture({200, 150}, 21);
    const cv::Rect box(60, 40, 41, 35);
    const cv::Point shift(17, -9);
    cv::Mat moved;
    const cv::Matx23d M(1, 0, shift.x, 0, 1, shift.y);
    cv::warpAffine(scene, moved, M, scene.size(), cv::INTER_NEAREST, cv::BORDER_REFLECT);
    GradientParams gp;
    const RTable t = buildRTable(orientations(scene, gp), box, gp.bins);
    ASSERT_GT(t.entries, 100);
    const cv::Mat votes = houghVotes(orientations(moved, gp), t);
    cv::Point arg;
    double peak = 0;
    cv::minMaxLoc(votes, nullptr, &peak, nullptr, &arg);
    EXPECT_EQ(arg, t.centre + shift);
    EXPECT_EQ(peak, double(t.entries));  // every model pixel finds its partner
    // A restricted search containing the answer finds the same peak.
    const cv::Mat local = houghVotes(orientations(moved, gp), t, {arg.x - 10, arg.y - 10, 21, 21});
    EXPECT_EQ(local.at<float>(arg), votes.at<float>(arg));
    EXPECT_EQ(cv::countNonZero(local(cv::Rect(0, 0, 20, 20))), 0);
}

TEST(Hough, OrientationOfARampAndItsBin) {
    // f(x, y) = 8 (x cos θ + y sin θ): the gradient is 8 (cos θ, sin θ) inside, whatever the smoothing
    // (up to the rounding to 8 bits), so the orientation must be θ on the whole circle, and the bin
    // ⌊(θ + π) · bins / 2π⌋ of the measured θ.
    GradientParams gp;
    gp.bins = 36;
    for (double theta = -3.1; theta < 3.14; theta += 0.37) {
        cv::Mat f(41, 41, CV_8U);
        for (int y = 0; y < f.rows; ++y)
            for (int x = 0; x < f.cols; ++x)
                f.at<uchar>(y, x) = cv::saturate_cast<uchar>(128 + 8.0 * ((x - 20) * std::cos(theta) + (y - 20) * std::sin(theta)));
        const Orientations o = orientations(f, gp);
        const double measured = o.angle.at<float>(20, 20);
        EXPECT_NEAR(o.magnitude.at<float>(20, 20), 8.0, 0.25) << theta;
        EXPECT_NEAR(measured, theta, 0.03) << theta;
        EXPECT_EQ(o.bin.at<ushort>(20, 20), int((measured + CV_PI) * gp.bins / (2 * CV_PI))) << theta;
        EXPECT_EQ(o.mask.at<uchar>(20, 20), 255) << theta;
    }
}

TEST(Hough, TracksTheSyntheticObject) {
    const auto s = vl::motion::test::noisyMovingObject({320, 240}, 120, 11);
    HoughTrackerParams p;
    p.searchRadius = 30;
    HoughTracker tr(p, s.frames[0], s.boxes[0]);
    std::vector<cv::Rect2d> est{tr.first().step.box};
    for (size_t t = 1; t < s.frames.size(); ++t) est.push_back(tr.step(s.frames[t]).step.box);
    const double m = meanIou(s, est);
    EXPECT_GT(m, 0.6) << "mean IoU " << m;
}

TEST(Kalman, PredictsAConstantVelocityTrackAndGetsSure) {
    KalmanParams kp;
    kp.processNoise = 0.05;
    kp.measurementNoise = 1.0;
    ConstantVelocityKalman kf(kp);
    const cv::Point2d p0(10, 20), v(3.0, -1.5);
    kf.init(p0);
    std::mt19937 rng(3);
    std::normal_distribution<double> n(0, 1.0);
    double prevTrace = 1e18;
    for (int t = 1; t <= 60; ++t) {
        kf.predict();
        const cv::Point2d truth = p0 + double(t) * v;
        if (t > 20) {  // after a few frames, the prediction alone is close to where the object is
            EXPECT_NEAR(kf.position().x, truth.x, 1.5) << t;
            EXPECT_NEAR(kf.position().y, truth.y, 1.5) << t;
        }
        kf.update(truth + cv::Point2d(n(rng), n(rng)));
        const cv::Matx22d P = kf.positionCovariance();
        const double tr = P(0, 0) + P(1, 1);
        if (t > 2) {
            EXPECT_LE(tr, prevTrace + 1e-9) << "frame " << t;
        }
        prevTrace = tr;
    }
    EXPECT_NEAR(kf.velocity().x, v.x, 0.2);
    EXPECT_NEAR(kf.velocity().y, v.y, 0.2);
    EXPECT_LT(prevTrace, 2 * 1.0);  // tighter than one raw measurement per axis
    // Coasting without measurements: the position keeps the velocity, the uncertainty grows.
    const cv::Point2d before = kf.position();
    for (int t = 0; t < 10; ++t) kf.predict();
    EXPECT_NEAR(kf.position().x, before.x + 10 * kf.velocity().x, 1e-9);
    const cv::Matx22d P = kf.positionCovariance();
    EXPECT_GT(P(0, 0) + P(1, 1), prevTrace * 2);
}

TEST(Kalman, CovarianceEllipseAxes) {
    const auto e = covarianceEllipse(cv::Matx22d(9, 0, 0, 4), {5, 6}, 2.0);
    EXPECT_NEAR(e.major, 6.0, 1e-9);
    EXPECT_NEAR(e.minor, 4.0, 1e-9);
    EXPECT_NEAR(e.angleDeg, 0.0, 1e-9);
    const auto r = covarianceEllipse(cv::Matx22d(5, 4, 4, 5), {0, 0}, 1.0);  // eigen 9 along (1, 1), 1 along (1, −1)
    EXPECT_NEAR(r.major, 3.0, 1e-9);
    EXPECT_NEAR(r.minor, 1.0, 1e-9);
    EXPECT_NEAR(r.angleDeg, 45.0, 1e-9);
}

TEST(Kalman, PredictionCarriesMeanShiftThroughAnOcclusion) {
    const auto s = occludedCrossing({320, 240}, 90, 5);
    auto run = [&](bool kalman) {
        MeanShiftTrackerParams p;
        p.usePrediction = kalman;
        MeanShiftTracker tr(p, s.frames[0], s.boxes[0]);
        std::vector<cv::Rect2d> est{tr.first().step.box};
        for (size_t t = 1; t < s.frames.size(); ++t) est.push_back(tr.step(s.frames[t]).step.box);
        // Score the frames after the pillar: has the tracker found the object again?
        double sum = 0;
        int n = 0;
        for (size_t t = est.size() * 3 / 4; t < est.size(); ++t, ++n) sum += iou(est[t], cv::Rect2d(s.boxes[t]));
        return sum / n;
    };
    const double without = run(false), with = run(true);
    EXPECT_LT(without, 0.2) << "plain mean shift should stay stuck at the pillar";
    EXPECT_GT(with, 0.6) << "with the prediction it should come out with the object";
}

}  // namespace

namespace {

TEST(Kalman, PriorKeepsHoughOnTheObjectThroughAnOcclusion) {
    using namespace vl::motion;
    const auto s = occludedCrossing({320, 240}, 90, 5);
    HoughTrackerParams p;
    p.usePrediction = true;
    p.searchRadius = 40;
    HoughTracker tr(p, s.frames[0], s.boxes[0]);
    double sum = 0;
    int n = 0, lostInside = 0;
    for (size_t t = 1; t < s.frames.size(); ++t) {
        const auto f = tr.step(s.frames[t]);
        if (t >= s.frames.size() * 3 / 4) {
            sum += iou(f.step.box, cv::Rect2d(s.boxes[t]));
            ++n;
        }
        // While the disc is fully hidden its votes vanish: the tracker must notice.
        if (cv::countNonZero(s.masks[t]) == 0) lostInside += f.step.lost;
    }
    EXPECT_GT(lostInside, 0);
    EXPECT_GT(sum / n, 0.6) << "mean IoU after the pillar";
}

}  // namespace
