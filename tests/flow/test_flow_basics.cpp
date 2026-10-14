// Building blocks of the flow area: derivatives, pyramids, warping, colour code, metrics, .flo I/O.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "core/flow/flow_color.hpp"
#include "core/flow/flow_common.hpp"
#include "core/flow/flow_eval.hpp"

using namespace vl::flow;

namespace {

cv::Mat ramp(cv::Size size, float a, float b) {  // I(x, y) = a·x + b·y
    cv::Mat m(size, CV_32F);
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x) m.at<float>(y, x) = a * float(x) + b * float(y);
    return m;
}

double hueDeg(const cv::Vec3b& bgr) {
    cv::Mat px(1, 1, CV_32FC3, cv::Scalar(bgr[0] / 255.0, bgr[1] / 255.0, bgr[2] / 255.0)), hsv;
    cv::cvtColor(px, hsv, cv::COLOR_BGR2HSV);
    return hsv.at<cv::Vec3f>(0, 0)[0];
}

}  // namespace

TEST(FlowDerivatives, ExactOnARampAndTemporalDifference) {
    const cv::Mat I0 = ramp({40, 30}, 3.f, -2.f);
    const cv::Mat I1 = I0 + 5.f;
    const Derivatives d = derivatives(I0, I1, true);
    // The 5-tap central difference is exact on polynomials up to degree 4, away from the border.
    for (int y = 2; y < 28; ++y)
        for (int x = 2; x < 38; ++x) {
            ASSERT_NEAR(d.Ix.at<float>(y, x), 3.f, 1e-4f);
            ASSERT_NEAR(d.Iy.at<float>(y, x), -2.f, 1e-4f);
            ASSERT_NEAR(d.It.at<float>(y, x), 5.f, 1e-4f);
        }
}

TEST(FlowDerivatives, AveragedOverBothFrames) {
    const cv::Mat I0 = ramp({20, 20}, 2.f, 0.f), I1 = ramp({20, 20}, 4.f, 0.f);
    EXPECT_NEAR(derivatives(I0, I1, true).Ix.at<float>(10, 10), 3.f, 1e-4f);
    EXPECT_NEAR(derivatives(I0, I1, false).Ix.at<float>(10, 10), 2.f, 1e-4f);
}

TEST(FlowPyramid, HalvesUntilTooSmall) {
    const auto pyr = gaussianPyramid(cv::Mat::zeros(100, 120, CV_32F), 6, 16);
    ASSERT_EQ(pyr.size(), 3u);  // 120x100 -> 60x50 -> 30x25; 15x13 would be under 16
    EXPECT_EQ(pyr[1].size(), cv::Size(60, 50));
    EXPECT_EQ(pyr[2].size(), cv::Size(30, 25));
}

TEST(FlowWarp, IntegerShiftAndOutsideMask) {
    cv::Mat img(20, 30, CV_32F);
    cv::randu(img, 0, 255);
    const cv::Mat flow(img.size(), CV_32FC2, cv::Scalar(2, -1));
    cv::Mat outside;
    const cv::Mat w = warpImage(img, flow, &outside);
    EXPECT_FLOAT_EQ(w.at<float>(5, 10), img.at<float>(4, 12));  // out(x) = img(x + w)
    EXPECT_EQ(outside.at<uchar>(0, 5), 255);                     // y - 1 < 0
    EXPECT_EQ(outside.at<uchar>(5, 28), 255);                    // x + 2 > 29
    EXPECT_EQ(outside.at<uchar>(5, 5), 0);
}

TEST(FlowWarp, ResizeScalesVectors) {
    const cv::Mat f(10, 20, CV_32FC2, cv::Scalar(1.5, -2));
    const cv::Mat g = resizeFlow(f, {40, 20});
    EXPECT_NEAR(g.at<cv::Vec2f>(7, 13)[0], 3.0f, 1e-5f);
    EXPECT_NEAR(g.at<cv::Vec2f>(7, 13)[1], -4.0f, 1e-5f);
}

TEST(FlowColor, AxisDirectionsHaveTheMiddleburyHues) {
    // Image coordinates (y down): right = red, down = yellow-orange, left = cyan, up = blue-violet.
    EXPECT_NEAR(hueDeg(flowColor(1, 0, 1)), 0.0, 8.0);
    const double down = hueDeg(flowColor(0, 1, 1));
    EXPECT_GT(down, 40.0);
    EXPECT_LT(down, 65.0);
    const double left = hueDeg(flowColor(-1, 0, 1));  // cyan-azure
    EXPECT_GT(left, 175.0);
    EXPECT_LT(left, 200.0);
    const double up = hueDeg(flowColor(0, -1, 1));
    EXPECT_GT(up, 245.0);
    EXPECT_LT(up, 280.0);
    // No motion is white; half the scale is half saturated; NaN is black.
    EXPECT_EQ(flowColor(0, 0, 1), cv::Vec3b(255, 255, 255));
    const cv::Vec3b half = flowColor(0.5f, 0, 1);
    EXPECT_EQ(half[2], 255);
    EXPECT_NEAR(half[1], 128, 2);
    EXPECT_EQ(flowColor(std::numeric_limits<float>::quiet_NaN(), 0, 1), cv::Vec3b(0, 0, 0));
}

TEST(FlowColor, WheelPositionInvertsTheKey) {
    const cv::Size size(200, 160);
    const cv::Mat wheel = colorWheel(size, 4.f);
    const cv::Point2f p = wheelPosition(size, 4.f, {2.f, -1.f});
    // The wheel is sampled at pixel centres, so allow the colour change across half a pixel.
    EXPECT_LT(cv::norm(cv::Vec3d(wheel.at<cv::Vec3b>(int(p.y), int(p.x))) - cv::Vec3d(flowColor(2.f, -1.f, 4.f))), 12.0);
}

TEST(FlowColor, ArrowsOnAGrid) {
    const cv::Mat f(64, 64, CV_32FC2, cv::Scalar(1, 0));
    const auto a = flowArrows(f, 16, 3.f);
    ASSERT_EQ(a.size(), 16u);
    EXPECT_FLOAT_EQ(a[0].to.x - a[0].from.x, 3.f);
    EXPECT_FLOAT_EQ(arrowScale(f, 16), 0.9f * 16);
}

TEST(FlowEval, EndpointAndAngularErrorOnAKnownField) {
    cv::Mat truth(10, 10, CV_32FC2, cv::Scalar(1, 2));
    truth.at<cv::Vec2f>(0, 0) = {1e10f, 1e10f};  // .flo "unknown": ignored
    cv::Mat flow = truth + cv::Scalar(3, 4);      // every known pixel is off by (3, 4): EPE = 5
    flow.at<cv::Vec2f>(0, 1) = {std::numeric_limits<float>::quiet_NaN(), 0.f};
    const FlowErrors e = evaluateFlow(flow, truth);
    EXPECT_EQ(e.count, 98);
    EXPECT_EQ(e.missing, 1);
    EXPECT_NEAR(e.epe, 5.0, 1e-6);
    EXPECT_DOUBLE_EQ(e.over1, 1.0);
    EXPECT_DOUBLE_EQ(e.over3, 1.0);
    EXPECT_TRUE(std::isnan(endpointErrorMap(flow, truth).at<float>(0, 0)));

    // Angle between (1, 0, 1) and (0, 0, 1) is 45°.
    const cv::Mat zero(4, 4, CV_32FC2, cv::Scalar(0, 0)), one(4, 4, CV_32FC2, cv::Scalar(1, 0));
    EXPECT_NEAR(evaluateFlow(one, zero).angular, 45.0, 1e-4);
    EXPECT_NEAR(angularErrorMap(zero, zero).at<float>(1, 1), 0.0, 1e-3);
}

TEST(FlowEval, ResidualIsZeroForTheTrueShift) {
    cv::Mat a(30, 40, CV_8U);
    cv::randu(a, 0, 255);
    cv::Mat b(a.size(), CV_8U, cv::Scalar(0));
    a(cv::Rect(0, 0, 37, 30)).copyTo(b(cv::Rect(3, 0, 37, 30)));  // b(x + 3) = a(x)
    const cv::Mat r = brightnessResidual(a, b, cv::Mat(a.size(), CV_32FC2, cv::Scalar(3, 0)));
    EXPECT_FLOAT_EQ(r.at<float>(10, 10), 0.f);
    EXPECT_TRUE(std::isnan(r.at<float>(10, 39)));  // sent outside frame 2
}

TEST(FlowEval, FloRoundTrip) {
    cv::Mat f(7, 11, CV_32FC2);
    cv::randu(f, -20, 20);
    f.at<cv::Vec2f>(3, 4) = {1e10f, 1e10f};
    const auto path = (std::filesystem::temp_directory_path() / "vl_flow_roundtrip.flo").string();
    writeFlo(path, f);
    const cv::Mat g = readFlo(path);
    ASSERT_EQ(g.size(), f.size());
    ASSERT_EQ(g.type(), CV_32FC2);
    EXPECT_EQ(cv::norm(f, g, cv::NORM_INF), 0.0);
    EXPECT_EQ(std::filesystem::file_size(path), 12u + 7u * 11u * 8u);
    EXPECT_FALSE(isKnownFlow(g.at<cv::Vec2f>(3, 4)));

    // A file without the "PIEH" tag is refused.
    {
        std::FILE* fp = std::fopen(path.c_str(), "wb");
        std::fputs("not a flow file", fp);
        std::fclose(fp);
    }
    EXPECT_THROW(readFlo(path), std::runtime_error);
    std::filesystem::remove(path);
}
