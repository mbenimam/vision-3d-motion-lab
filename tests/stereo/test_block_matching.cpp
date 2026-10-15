#include <cmath>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "core/common/synthetic.hpp"
#include "core/stereo/block_matching.hpp"

namespace {

using vl::stereo::blockMatch;
using vl::stereo::BlockMatchParams;
using vl::stereo::Cost;

TEST(BoxSum, MatchesBruteForceIncludingBorders) {
    cv::Mat f(17, 23, CV_32F);
    cv::randu(f, -5, 5);
    const int r = 3;
    const cv::Mat s = vl::stereo::boxSum(f, r);
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x) {
            double ref = 0;
            for (int v = std::max(0, y - r); v <= std::min(f.rows - 1, y + r); ++v)
                for (int u = std::max(0, x - r); u <= std::min(f.cols - 1, x + r); ++u) ref += f.at<float>(v, u);
            ASSERT_NEAR(s.at<float>(y, x), ref, 1e-3) << "at " << x << "," << y;
        }
}

// A pure shift: every pixel has the same disparity, so every cost must find it exactly.
TEST(BlockMatch, RecoversAConstantShiftWithEveryCost) {
    const cv::Mat tex = vl::synth::texture({160, 120}, 3);
    const int shift = 9;
    cv::Mat right = cv::Mat::zeros(tex.size(), tex.type());
    tex.colRange(shift, tex.cols).copyTo(right.colRange(0, tex.cols - shift));  // left(x) == right(x - 9)
    for (Cost c : {Cost::SAD, Cost::SSD, Cost::ZNCC}) {
        BlockMatchParams p;
        p.cost = c;
        p.maxDisparity = 20;
        p.leftRightCheck = false;
        const auto res = blockMatch(tex, right, p);
        int good = 0, total = 0;
        for (int y = 5; y < tex.rows - 5; ++y)
            for (int x = shift + 10; x < tex.cols - 5; ++x, ++total)
                good += std::abs(res.disparity.at<float>(y, x) - shift) < 0.5f;
        EXPECT_GT(good, 0.99 * total) << "cost " << int(c);
    }
}

// ZNCC ignores a gain and offset between the cameras; SSD does not.
TEST(BlockMatch, ZnccSurvivesABrightnessChange) {
    const cv::Mat tex = vl::synth::texture({160, 120}, 5);
    const int shift = 6;
    cv::Mat right = cv::Mat::zeros(tex.size(), tex.type());
    tex.colRange(shift, tex.cols).copyTo(right.colRange(0, tex.cols - shift));
    right.convertTo(right, -1, 0.6, 60);  // darker gain, lifted offset
    auto rate = [&](Cost c) {
        BlockMatchParams p;
        p.cost = c;
        p.maxDisparity = 16;
        p.leftRightCheck = false;
        const auto res = blockMatch(tex, right, p);
        int good = 0, total = 0;
        for (int y = 5; y < tex.rows - 5; ++y)
            for (int x = shift + 10; x < tex.cols - 5; ++x, ++total)
                good += std::abs(res.disparity.at<float>(y, x) - shift) < 0.5f;
        return double(good) / total;
    };
    EXPECT_GT(rate(Cost::ZNCC), 0.98);
    EXPECT_LT(rate(Cost::SSD), 0.8);
}

// On the layered synthetic scene: low error on visible pixels, and the left-right check
// rejects mostly the occluded band to the left of each near object.
TEST(BlockMatch, LayeredSceneAccuracyAndOcclusionRejection) {
    const auto s = vl::synth::layeredStereo({256, 192}, 24, 11);
    BlockMatchParams p;
    p.maxDisparity = 28;
    p.radius = 3;
    const auto res = blockMatch(s.left, s.right, p);
    cv::Mat visible;
    cv::bitwise_not(s.occluded, visible);
    EXPECT_LT(vl::stereo::badPixelRate(res.disparity, s.disparity, visible, 1.0), 0.10);
    const double rejectedInOccluded = cv::countNonZero(res.rejected & s.occluded) / double(cv::countNonZero(s.occluded));
    const double rejectedInVisible = cv::countNonZero(res.rejected & visible) / double(cv::countNonZero(visible));
    EXPECT_GT(rejectedInOccluded, 0.5);
    EXPECT_LT(rejectedInVisible, 0.10);
}

TEST(BlockMatch, StopsWhenAsked) {
    const auto s = vl::synth::layeredStereo({128, 96}, 16, 2);
    std::stop_source src;
    src.request_stop();
    EXPECT_THROW(blockMatch(s.left, s.right, {}, src.get_token()), vl::Cancelled);
}

TEST(BadPixelRate, CountsRejectedPixelsAsErrorsAndSkipsUnknownTruth) {
    float dv[] = {1.f, 5.f, std::nanf(""), 2.f};
    float tv[] = {1.f, 2.f, 3.f, 0.f};  // last pixel: unknown truth
    const cv::Mat d(1, 4, CV_32F, dv), t(1, 4, CV_32F, tv);
    EXPECT_NEAR(vl::stereo::badPixelRate(d, t, cv::Mat(), 1.0), 2.0 / 3.0, 1e-9);
}

}  // namespace
