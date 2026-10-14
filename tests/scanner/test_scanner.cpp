// Document scanner: each stage against exact ground truth, and the whole pipeline on every synthetic photo.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "core/common/cv_compat.hpp"
#include "core/scanner/opencv_baselines.hpp"
#include "core/scanner/scanner.hpp"
#include "core/scanner/synthetic_page.hpp"

using namespace vl::scanner;

namespace {

const DeskScene kScenes[] = {DeskScene::Easy, DeskScene::StrongPerspective, DeskScene::Shadowed, DeskScene::LowContrast,
                             DeskScene::Cluttered};

double maxCornerError(const Quad& a, const Quad& b) {
    double worst = 0;
    for (size_t i = 0; i < 4; ++i) worst = std::max(worst, std::hypot(a[i].x - b[i].x, a[i].y - b[i].y));
    return worst;
}

}  // namespace

// --------------------------------------------------------------------------- stages

TEST(Canny, ThinEdgeOnTheBorderOfASquare) {
    cv::Mat f(120, 120, CV_32F, cv::Scalar(40));
    f(cv::Rect(30, 30, 60, 60)).setTo(200);  // step between columns 29 and 30, rows 29 and 30
    const EdgeMap e = canny(f, CannyParams{});
    int onBorder = 0, total = 0;
    for (int y = 0; y < 120; ++y)
        for (int x = 0; x < 120; ++x) {
            if (!e.edges.at<uchar>(y, x)) continue;
            ++total;
            const bool nearVertical = (std::abs(x - 29.5) <= 1 || std::abs(x - 89.5) <= 1) && y > 35 && y < 85;
            const bool nearHorizontal = (std::abs(y - 29.5) <= 1 || std::abs(y - 89.5) <= 1) && x > 35 && x < 85;
            onBorder += nearVertical || nearHorizontal;
        }
    // Thin: along the middle of the left side, exactly one edge pixel per row.
    for (int y = 40; y < 80; ++y) {
        int n = 0;
        for (int x = 20; x < 40; ++x) n += e.edges.at<uchar>(y, x) != 0;
        EXPECT_EQ(n, 1) << "row " << y;
    }
    EXPECT_GE(onBorder, 4 * 45);
    EXPECT_LT(total, 4 * 64);  // nothing but the border
}

TEST(Hough, FindsTheFourSidesOfARectangle) {
    cv::Mat f(200, 240, CV_32F, cv::Scalar(30));
    f(cv::Rect(50, 40, 120, 100)).setTo(220);
    const EdgeMap e = canny(f, CannyParams{});
    const HoughResult h = houghLines(e.edges, e.dx, e.dy, HoughParams{});
    ASSERT_GE(h.lines.size(), 4u);
    // Expected: x = 49.5 and 169.5 (θ = 0), y = 39.5 and 139.5 (θ = 90°), within the bins' 1 px / 1°.
    const double expected[4][2] = {{0, 49.5}, {0, 169.5}, {90, 39.5}, {90, 139.5}};
    for (const auto& ex : expected) {
        bool found = false;
        for (size_t i = 0; i < 4; ++i) {
            const double deg = h.lines[i].theta * 180 / std::numbers::pi;
            found = found || (std::abs(deg - ex[0]) <= 1 && std::abs(h.lines[i].rho - ex[1]) <= 1.5);
        }
        EXPECT_TRUE(found) << "θ = " << ex[0] << ", ρ = " << ex[1];
    }
}

TEST(Hough, IntersectionOfTwoLines) {
    cv::Point2d p;
    ASSERT_TRUE(intersect({0, 10, 0}, {std::numbers::pi / 2, 20, 0}, p));  // x = 10 meets y = 20
    EXPECT_NEAR(p.x, 10, 1e-9);
    EXPECT_NEAR(p.y, 20, 1e-9);
    EXPECT_FALSE(intersect({0.3, 10, 0}, {0.3, 40, 0}, p));
}

TEST(Corners, OrderedWhateverTheInputOrder) {
    const Quad truth = {cv::Point2d(100, 80), cv::Point2d(400, 110), cv::Point2d(380, 500), cv::Point2d(90, 470)};
    std::vector<int> perm = {0, 1, 2, 3};
    do {
        std::vector<cv::Point2d> shuffled;
        for (int i : perm) shuffled.push_back(truth[size_t(i)]);
        const Quad q = orderCorners(shuffled);
        for (size_t i = 0; i < 4; ++i) EXPECT_EQ(q[i], truth[i]);
    } while (std::next_permutation(perm.begin(), perm.end()));
    // A page turned by 30°: still clockwise, starting from the corner nearest the photo's top-left.
    const Quad turned = orderCorners({{300, 50}, {520, 430}, {170, 630}, {-50, 250}});
    EXPECT_EQ(turned[0], cv::Point2d(-50, 250));
    EXPECT_EQ(turned[1], cv::Point2d(300, 50));
    EXPECT_EQ(turned[2], cv::Point2d(520, 430));
    EXPECT_EQ(turned[3], cv::Point2d(170, 630));
}

TEST(Rectify, HomographyMapsTheCornersExactly) {
    const Quad corners = {cv::Point2d(262, 250), cv::Point2d(712, 236), cv::Point2d(908, 1156), cv::Point2d(66, 1140)};
    const cv::Size size(600, 849);
    const cv::Matx33d H = pageHomography(corners, size);
    const Quad target = rectangleCorners(size);
    for (size_t i = 0; i < 4; ++i) {
        const cv::Vec3d x = H * cv::Vec3d(corners[i].x, corners[i].y, 1);
        EXPECT_NEAR(x[0] / x[2], target[i].x, 1e-6);
        EXPECT_NEAR(x[1] / x[2], target[i].y, 1e-6);
    }
    // The same map as OpenCV's 4-point solution (both scaled so H(2,2) = 1).
    std::vector<cv::Point2f> src, dst;
    for (size_t i = 0; i < 4; ++i) {
        src.emplace_back(corners[i]);
        dst.emplace_back(target[i]);
    }
    const cv::Matx33d ref(cv::getPerspectiveTransform(src, dst));
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(H(r, c) / H(2, 2), ref(r, c) / ref(2, 2), 1e-4 * (1 + std::abs(ref(r, c))));
}

TEST(Rectify, OutputSizeFromSidesAndA4) {
    const Quad q = {cv::Point2d(0, 0), cv::Point2d(300, 0), cv::Point2d(310, 420), cv::Point2d(0, 400)};
    // W = max(top 300, bottom √(310² + 20²) = 310.6), H = max(left 400, right √(10² + 420²) = 420.1).
    EXPECT_EQ(outputSize(q, PageRatio::Auto), cv::Size(311, 420));
    const cv::Size a4 = outputSize(q, PageRatio::A4);
    EXPECT_EQ(a4.height, 420);
    EXPECT_NEAR(double(a4.height) / a4.width, std::sqrt(2.0), 0.01);
}

TEST(Cleanup, IntegralImageGivesWindowSums) {
    cv::Mat f(17, 23, CV_32F);
    cv::randu(f, 0, 255);
    const cv::Mat S = integralImage(f);
    EXPECT_NEAR(S.at<double>(17, 23), cv::sum(f)[0], 1e-6 * cv::sum(f)[0]);
    const cv::Rect w(4, 5, 9, 7);
    const double viaS = S.at<double>(12, 13) - S.at<double>(5, 13) - S.at<double>(12, 4) + S.at<double>(5, 4);
    EXPECT_NEAR(viaS, cv::sum(f(w))[0], 1e-3);
}

// The window mean from the integral image equals the mean computed pixel by pixel, borders included.
TEST(Cleanup, AdaptiveThresholdMatchesABruteForceWindowMean) {
    cv::Mat f(40, 50, CV_32F);
    cv::randu(f, 0, 255);
    const int r = 4;
    const double C = 7;
    const cv::Mat bw = adaptiveThreshold(f, r, C);
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x) {
            double sum = 0;
            int n = 0;
            for (int j = std::max(0, y - r); j <= std::min(f.rows - 1, y + r); ++j)
                for (int i = std::max(0, x - r); i <= std::min(f.cols - 1, x + r); ++i, ++n) sum += f.at<float>(j, i);
            ASSERT_EQ(bw.at<uchar>(y, x), f.at<float>(y, x) >= sum / n - C ? 255 : 0) << x << ", " << y;
        }
}

// Dark squares 40 levels below a background that ramps from 60 to 230: any single threshold either
// misses the squares on the bright side or blackens the dark side; the local mean minus C recovers them.
TEST(Cleanup, AdaptiveThresholdRecoversInkOnAGradient) {
    cv::Mat f(200, 400, CV_32F), ink = cv::Mat::zeros(200, 400, CV_8U);
    for (int y = 0; y < 200; ++y)
        for (int x = 0; x < 400; ++x) f.at<float>(y, x) = float(60 + 170 * x / 399.0);
    for (int x0 = 20; x0 < 380; x0 += 60) {
        const cv::Rect r(x0, 80, 8, 40);
        f(r) -= 40;
        ink(r).setTo(255);
    }
    const cv::Mat bw = adaptiveThreshold(f, 15, 10);
    int wrong = 0;
    for (int y = 0; y < 200; ++y)
        for (int x = 0; x < 400; ++x) wrong += (bw.at<uchar>(y, x) == 0) != (ink.at<uchar>(y, x) != 0);
    EXPECT_EQ(wrong, 0);
    // The global mean as a threshold gets a large share wrong.
    const double mean = cv::mean(f)[0];
    int globalWrong = 0;
    for (int y = 0; y < 200; ++y)
        for (int x = 0; x < 400; ++x) globalWrong += (f.at<float>(y, x) < mean) != (ink.at<uchar>(y, x) != 0);
    EXPECT_GT(globalWrong, 200 * 400 / 4);
}

TEST(Cleanup, IlluminationFollowsAShadowNotTheText) {
    // Paper at 230 under a light that falls to 50 % on the right, with black text strokes everywhere.
    cv::Mat page(300, 400, CV_32F);
    for (int y = 0; y < 300; ++y)
        for (int x = 0; x < 400; ++x) {
            const double light = 1.0 - 0.5 * x / 399.0;
            const bool stroke = (y % 20 < 3) && (x % 50 < 40);
            page.at<float>(y, x) = float((stroke ? 30 : 230) * light);
        }
    const cv::Mat L = estimateIllumination(page);
    for (int x : {20, 200, 380}) EXPECT_NEAR(L.at<float>(150, x), 230 * (1 - 0.5 * x / 399.0), 12) << "x = " << x;
}

// --------------------------------------------------------------------------- whole pipeline

TEST(Scanner, CornersWithinAPixelOnEverySyntheticPhoto) {
    for (const DeskScene s : kScenes) {
        const DocumentPhoto d = documentPhoto(s);
        const PageDetection det = findPage(d.photo, ScanParams{});
        ASSERT_TRUE(det.found) << int(s);
        EXPECT_LT(maxCornerError(det.corners, d.corners), 1.5) << "scene " << int(s);
        EXPECT_LT(cornerError(det.corners, d.corners), 0.8) << "scene " << int(s);
    }
}

TEST(Scanner, RefinementMakesTheCornersSubPixel) {
    const DocumentPhoto d = documentPhoto(DeskScene::Easy);
    ScanParams coarse;
    coarse.refineSides = false;
    const double before = cornerError(findPage(d.photo, coarse).corners, d.corners);
    const double after = cornerError(findPage(d.photo, ScanParams{}).corners, d.corners);
    EXPECT_LT(after, 0.6);
    EXPECT_LT(after, before);
}

TEST(Scanner, RectifiedPageIsCloseToTheOriginal) {
    for (const DeskScene s : {DeskScene::Easy, DeskScene::Shadowed}) {
        const DocumentPhoto d = documentPhoto(s);
        const ScanResult r = scanDocument(d.photo, ScanParams{});
        // Geometry: the page unwarped with our corners vs with the true corners.
        const cv::Mat exact = rectify(d.photo, d.corners, d.page.size());
        EXPECT_GT(psnr(rectify(d.photo, r.corners, d.page.size()), exact), 36.0) << int(s);
        // Content: the black-and-white scan of the page, at the original page's size, against the
        // original's ink. Within 1 px (resampling moves stroke borders by up to half a pixel), the ink
        // found must be true ink (precision) and the true ink must be found (recall).
        CleanupParams bw;
        bw.mode = ScanMode::BlackWhite;
        const cv::Mat scan = cleanUp(rectify(d.photo, r.corners, d.page.size()), bw).scan;
        cv::Mat grey;
        cv::cvtColor(d.page, grey, cv::COLOR_BGR2GRAY);
        const cv::Mat inkTrue = grey < 150, inkScan = scan < 128;
        cv::Mat nearTrue, nearScan;
        cv::dilate(inkTrue, nearTrue, cv::Mat::ones(3, 3, CV_8U));
        cv::dilate(inkScan, nearScan, cv::Mat::ones(3, 3, CV_8U));
        const double precision = double(cv::countNonZero(inkScan & nearTrue)) / cv::countNonZero(inkScan);
        const double recall = double(cv::countNonZero(inkTrue & nearScan)) / cv::countNonZero(inkTrue);
        std::printf("scene %d: ink precision %.3f, recall %.3f\n", int(s), precision, recall);
        EXPECT_GT(precision, 0.92) << int(s);  // thresholding a blurred photo thickens strokes a little
        EXPECT_GT(recall, 0.98) << int(s);
        // Shadow removed: the paper of the scan is uniformly white (median of the page = 255).
        cv::Mat colour = r.clean.scan;
        cv::Mat g;
        cv::cvtColor(colour, g, cv::COLOR_BGR2GRAY);
        EXPECT_GT(cv::mean(g)[0], 215) << int(s);
    }
}

TEST(Scanner, GivenCornersAreUsedAsIs) {
    const DocumentPhoto d = documentPhoto(DeskScene::Easy);
    Quad shuffled = {d.corners[2], d.corners[0], d.corners[3], d.corners[1]};
    const ScanResult r = scanDocument(d.photo, ScanParams{}, shuffled);
    EXPECT_TRUE(r.manual);
    EXPECT_LT(maxCornerError(r.corners, d.corners), 1e-9);
}

TEST(Scanner, BeatsTheContourRecipeWhenACornerIsHidden) {
    const DocumentPhoto d = documentPhoto(DeskScene::Cluttered);
    const ScanParams p;
    const PageDetection det = findPage(d.photo, p);
    const auto o = opencv::documentScan(d.photo, p.detectionSide, det.edges.low, det.edges.high, {600, 849}, p.cleanup);
    EXPECT_LT(cornerError(det.corners, d.corners), 1.0);
    EXPECT_GT(cornerError(o.corners, d.corners), 20.0);
}
