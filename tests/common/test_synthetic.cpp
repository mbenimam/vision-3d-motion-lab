#include <cmath>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "core/common/image.hpp"
#include "core/common/synthetic.hpp"

namespace {

using namespace vl::synth;

// The stereo ground truth must be exact: a visible left pixel x shows the same surface point
// as the right pixel x - d (up to the sensor noise added to both views).
TEST(LayeredStereo, VisiblePixelsMatchTheirPartner) {
    const auto s = layeredStereo({200, 150}, 20, 4);
    double err = 0;
    long n = 0;
    for (int y = 0; y < s.left.rows; ++y)
        for (int x = 0; x < s.left.cols; ++x) {
            if (s.occluded.at<uchar>(y, x)) continue;
            const int xr = int(std::lround(float(x) - s.disparity.at<float>(y, x)));
            if (xr < 0) continue;
            const cv::Vec3b a = s.left.at<cv::Vec3b>(y, x), b = s.right.at<cv::Vec3b>(y, xr);
            for (int c = 0; c < 3; ++c) err += std::abs(int(a[c]) - int(b[c]));
            n += 3;
        }
    EXPECT_LT(err / double(n), 12.0);  // noise (σ = 2 per view) plus rounding of the slanted wall
}

TEST(LayeredStereo, OcclusionsSitLeftOfNearObjects) {
    const auto s = layeredStereo({200, 150}, 20, 4);
    const double share = cv::countNonZero(s.occluded) / double(s.occluded.total());
    EXPECT_GT(share, 0.02);
    EXPECT_LT(share, 0.25);
}

TEST(RigidFlow, Frame1IsFrame0MovedByTheFlow) {
    const auto p = rigidFlow({160, 120}, 3.0, 1.02, {2.0, -1.5}, 9);
    // Sample frame1 where each frame0 pixel lands; it must look like frame0 there.
    double err = 0;
    long n = 0;
    for (int y = 10; y < 110; y += 3)
        for (int x = 10; x < 150; x += 3) {
            const auto f = p.flow.at<cv::Vec2f>(y, x);
            const int x1 = int(std::lround(float(x) + f[0])), y1 = int(std::lround(float(y) + f[1]));
            if (x1 < 0 || y1 < 0 || x1 >= 160 || y1 >= 120) continue;
            const cv::Vec3b a = p.frame0.at<cv::Vec3b>(y, x), b = p.frame1.at<cv::Vec3b>(y1, x1);
            for (int c = 0; c < 3; ++c) err += std::abs(int(a[c]) - int(b[c]));
            n += 3;
        }
    EXPECT_LT(err / double(n), 10.0);
}

TEST(MovingObject, MasksAndBoxesAgree) {
    const auto m = movingObject({160, 120}, 30, 1);
    ASSERT_EQ(m.frames.size(), 30u);
    for (size_t t = 0; t < m.frames.size(); ++t) {
        std::vector<cv::Point> on;
        cv::findNonZero(m.masks[t], on);
        ASSERT_FALSE(on.empty());
        cv::Rect b(on.front(), cv::Size(1, 1));
        for (const auto& q : on) b |= cv::Rect(q, cv::Size(1, 1));
        EXPECT_LE((b & m.boxes[t]).area(), m.boxes[t].area());
        EXPECT_GT((b & m.boxes[t]).area(), 0.8 * b.area()) << "frame " << t;
    }
}

TEST(Image, ColorizeBlacksOutInvalidPixels) {
    float values[] = {0.f, 5.f, 10.f};
    uchar mask[] = {255, 0, 255};
    const cv::Mat v(1, 3, CV_32F, values), valid(1, 3, CV_8U, mask);
    const cv::Mat c = vl::colorize(v, 0, 10, valid);
    EXPECT_EQ(c.at<cv::Vec3b>(0, 1), cv::Vec3b(0, 0, 0));
    EXPECT_NE(c.at<cv::Vec3b>(0, 0), c.at<cv::Vec3b>(0, 2));
}

}  // namespace
