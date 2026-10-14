// Our filters against OpenCV's, which serve as the reference implementation.
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "core/common/filters.hpp"
#include "core/common/image.hpp"
#include "core/common/synthetic.hpp"

namespace {

cv::Mat testImage() { return vl::toGrayF(vl::synth::texture({96, 72}, 5)); }

double maxDiff(const cv::Mat& a, const cv::Mat& b) { return cv::norm(a, b, cv::NORM_INF); }

TEST(Filters, GaussianBlurMatchesOpenCv) {
    const cv::Mat f = testImage();
    for (double sigma : {0.8, 1.5, 3.0}) {
        cv::Mat ref;
        const int r = int(std::ceil(3 * sigma));
        cv::GaussianBlur(f, ref, cv::Size(2 * r + 1, 2 * r + 1), sigma, sigma, cv::BORDER_REPLICATE);
        EXPECT_LT(maxDiff(vl::gaussianBlur(f, sigma), ref), 0.05) << "sigma " << sigma;
    }
}

TEST(Filters, SeparableEqualsTheFull2dKernel) {
    const cv::Mat f = testImage();
    const auto g = vl::gaussianKernel(1.2);
    cv::Mat k(int(g.size()), int(g.size()), CV_32F);
    for (int j = 0; j < k.rows; ++j)
        for (int i = 0; i < k.cols; ++i) k.at<float>(j, i) = g[size_t(j)] * g[size_t(i)];
    EXPECT_LT(maxDiff(vl::convolveSeparable(f, g, g), vl::convolve(f, k)), 1e-3);
}

TEST(Filters, SobelIsScaledToTheSlope) {
    cv::Mat ramp(20, 30, CV_32F);
    for (int y = 0; y < ramp.rows; ++y)
        for (int x = 0; x < ramp.cols; ++x) ramp.at<float>(y, x) = 3.f * float(x) - 2.f * float(y);
    const auto g = vl::sobel(ramp);
    EXPECT_NEAR(g.dx.at<float>(10, 15), 3.f, 1e-4);
    EXPECT_NEAR(g.dy.at<float>(10, 15), -2.f, 1e-4);
    // Same as OpenCV's Sobel divided by 8.
    const cv::Mat f = testImage();
    cv::Mat ref;
    cv::Sobel(f, ref, CV_32F, 1, 0, 3, 1.0 / 8.0, 0, cv::BORDER_REPLICATE);
    EXPECT_LT(maxDiff(vl::sobel(f).dx, ref), 1e-3);
}

TEST(Filters, MorphologyMatchesOpenCvWithADisc) {
    cv::Mat mask(64, 64, CV_8U);
    cv::randu(mask, 0, 2);
    mask *= 255;
    const cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat refOpen;
    cv::morphologyEx(mask, refOpen, cv::MORPH_OPEN, se, {-1, -1}, 1, cv::BORDER_REPLICATE);
    // OpenCV's 5×5 ellipse is our disc of radius 2.
    EXPECT_EQ(cv::countNonZero(vl::open(mask, 2) != refOpen), 0);
}

TEST(Filters, BilinearInterpolatesBetweenPixels) {
    float v[] = {0.f, 10.f, 20.f, 30.f};
    const cv::Mat f(2, 2, CV_32F, v);
    EXPECT_FLOAT_EQ(vl::bilinear(f, 0.5f, 0.5f), 15.f);
    EXPECT_FLOAT_EQ(vl::bilinear(f, 1.f, 0.f), 10.f);
}

TEST(Filters, PyramidHalvesTheSize) {
    const auto p = vl::gaussianPyramid(testImage(), 3);
    ASSERT_EQ(p.size(), 3u);
    EXPECT_EQ(p[1].size(), cv::Size(48, 36));
    EXPECT_EQ(p[2].size(), cv::Size(24, 18));
}

}  // namespace
