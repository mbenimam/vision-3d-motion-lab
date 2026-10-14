#include <thread>

#include <gtest/gtest.h>

#include "core/geometry/features.hpp"
#include "core/geometry/homography.hpp"
#include "core/geometry/scenes.hpp"
#include "core/geometry/stitch.hpp"

using namespace vl::geometry;

TEST(Stitch, ChainComposesLinks) {
    const cv::Matx33d A(1, 0, 100, 0, 1, 5, 0, 0, 1), B(1.1, 0, 30, 0, 1.1, -4, 1e-4, 0, 1);
    const auto toRef = chainToReference({A, B}, 1);
    ASSERT_EQ(toRef.size(), 3u);
    EXPECT_LT(cv::norm(toRef[0] - A), 1e-12);
    EXPECT_LT(cv::norm(toRef[1] - cv::Matx33d::eye()), 1e-12);
    const cv::Matx33d Binv = B.inv() * (1.0 / B.inv()(2, 2));
    EXPECT_LT(cv::norm(toRef[2] - Binv), 1e-9);
    const auto toFirst = chainToReference({A, B}, 0);
    const cv::Point2d p(17, 23);
    const cv::Point2d q = transform(toFirst[2], transform(B, transform(A, p)));
    EXPECT_NEAR(cv::norm(q - p), 0, 1e-9);
}

TEST(Stitch, CanvasBoundsCoverEveryWarpedCorner) {
    // Reference 400x300 at the origin, a second image shifted right by 250 px and down by 40.
    const std::vector<cv::Size> sizes = {{400, 300}, {400, 300}};
    const std::vector<cv::Matx33d> toRef = {cv::Matx33d::eye(), cv::Matx33d(1, 0, 250, 0, 1, 40, 0, 0, 1)};
    const cv::Rect2d b = warpedBounds(sizes, toRef);
    EXPECT_NEAR(b.x, 0, 1e-9);
    EXPECT_NEAR(b.y, 0, 1e-9);
    EXPECT_NEAR(b.width, 650, 1e-9);
    EXPECT_NEAR(b.height, 340, 1e-9);

    cv::Mat img(300, 400, CV_8UC3, cv::Scalar(10, 200, 30));
    const auto pano = stitch({img, img}, toRef);
    EXPECT_EQ(pano.image.cols, 650);
    EXPECT_EQ(pano.image.rows, 340);
    // Covered pixels carry the colour, uncovered ones (bottom-left, top-right) stay black.
    EXPECT_EQ(pano.image.at<cv::Vec3b>(100, 100), cv::Vec3b(10, 200, 30));
    EXPECT_EQ(pano.image.at<cv::Vec3b>(330, 50), cv::Vec3b(0, 0, 0));
    EXPECT_EQ(pano.image.at<cv::Vec3b>(10, 640), cv::Vec3b(0, 0, 0));
}

TEST(Stitch, NegativeOffsetsShiftTheCanvas) {
    const std::vector<cv::Size> sizes = {{200, 100}, {200, 100}};
    const std::vector<cv::Matx33d> toRef = {cv::Matx33d(1, 0, -150, 0, 1, -20, 0, 0, 1), cv::Matx33d::eye()};
    const cv::Mat img(100, 200, CV_8UC3, cv::Scalar(255, 255, 255));
    const auto pano = stitch({img, img}, toRef);
    EXPECT_EQ(pano.image.size(), cv::Size(350, 120));
    // The reference's top-left corner lands at (150, 20) on the canvas.
    const cv::Point2d c = transform(pano.toCanvas[1], {0, 0});
    EXPECT_NEAR(c.x, 150, 1e-9);
    EXPECT_NEAR(c.y, 20, 1e-9);
}

TEST(Stitch, FeatherHidesExposureSeam) {
    // Two flat images of different brightness overlapping by half: with feathering the overlap ramps
    // smoothly, without it the step is a single column.
    const cv::Mat dark(100, 200, CV_8UC3, cv::Scalar::all(80)), bright(100, 200, CV_8UC3, cv::Scalar::all(180));
    const std::vector<cv::Matx33d> toRef = {cv::Matx33d::eye(), cv::Matx33d(1, 0, 100, 0, 1, 0, 0, 0, 1)};
    auto maxStep = [](const cv::Mat& im) {
        int m = 0;
        for (int x = 1; x < im.cols; ++x) m = std::max(m, std::abs(int(im.at<cv::Vec3b>(50, x)[0]) - im.at<cv::Vec3b>(50, x - 1)[0]));
        return m;
    };
    EXPECT_GE(maxStep(stitch({dark, bright}, toRef, {false}).image), 90);
    EXPECT_LE(maxStep(stitch({dark, bright}, toRef, {true}).image), 10);
}

TEST(Stitch, SyntheticRotatingCameraEndToEnd) {
    // SIFT + ratio test + RANSAC on the synthetic views must recover the exact homographies.
    const auto views = scenes::rotatingCamera({480, 360}, 3, 24.0, 4);
    std::vector<cv::Matx33d> links;
    for (int i = 0; i + 1 < 3; ++i) {
        const auto fa = detectSift(views.images[size_t(i)], 2000), fb = detectSift(views.images[size_t(i + 1)], 2000);
        const auto m = matchDescriptors(fa.descriptors, fb.descriptors, {0.75f, true});
        ASSERT_GT(m.size(), 60u);
        Points a, b;
        for (const auto& mm : m) a.emplace_back(fa.keypoints[size_t(mm.a)].pt), b.emplace_back(fb.keypoints[size_t(mm.b)].pt);
        const auto e = estimateHomography(a, b, {});
        ASSERT_TRUE(e.ok);
        EXPECT_GT(double(e.nInliers) / double(m.size()), 0.8);
        links.push_back(e.H);
    }
    const auto toRef = chainToReference(links, views.reference);
    for (int i = 0; i < 3; ++i) EXPECT_LT(cornerError(toRef[size_t(i)], views.toReference[size_t(i)], {480, 360}), 1.5);
}

// The app builds samples on worker threads: a scene must not depend on which thread built it
// (OpenCV's global RNG is per thread, so any unseeded noise would break this).
TEST(Scenes, RotatingCameraIsTheSameOnEveryThread) {
    const auto here = scenes::rotatingCamera({160, 120}, 2, 20.0, 4);
    scenes::RotatingViews there;
    std::thread([&] {
        cv::theRNG().next();  // a worker whose global RNG has already been used
        there = scenes::rotatingCamera({160, 120}, 2, 20.0, 4);
    }).join();
    ASSERT_EQ(here.images.size(), there.images.size());
    for (size_t i = 0; i < here.images.size(); ++i)
        EXPECT_EQ(cv::norm(here.images[i], there.images[i], cv::NORM_INF), 0.0) << "view " << i;
}

TEST(Matching, MutualCheckAndRatioFilter) {
    // Descriptors b = a permuted plus noise, and one duplicated row in b that makes the ratio test fail.
    cv::Mat a(50, 128, CV_32F);
    cv::randu(a, 0, 1);
    cv::Mat b(51, 128, CV_32F);
    for (int i = 0; i < 50; ++i) a.row(49 - i).copyTo(b.row(i));
    a.row(0).copyTo(b.row(50));  // a[0] now has two identical neighbours (b[49] and b[50])
    const auto m = matchDescriptors(a, b, {0.8f, true});
    EXPECT_EQ(m.size(), 49u);
    for (const auto& mm : m) {
        EXPECT_EQ(mm.b, 49 - mm.a);
        EXPECT_NE(mm.a, 0);
    }
    // With both filters off every descriptor keeps its nearest neighbour.
    const auto all = matchDescriptors(a, b, {1.0f, false});
    EXPECT_EQ(all.size(), 50u);
}
