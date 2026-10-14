#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "core/common/cv_compat.hpp"
#include "core/geometry/homography.hpp"
#include "core/geometry/matching.hpp"
#include "core/geometry/opencv_baselines.hpp"
#include "core/geometry/scenes.hpp"
#include "core/geometry/sift.hpp"

using namespace vl::geometry;

namespace {

// A grey image with one Gaussian blob of standard deviation t centred at (cx, cy), drawn analytically.
cv::Mat blobImage(cv::Size size, double cx, double cy, double t) {
    cv::Mat img(size, CV_8U);
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x) {
            const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            img.at<uchar>(y, x) = cv::saturate_cast<uchar>(40 + 180 * std::exp(-r2 / (2 * t * t)));
        }
    return img;
}

// Fraction of the points of a that, mapped by H, land inside `sizeB` and have a point of b within eps.
double repeatability(const std::vector<cv::Point2d>& a, const std::vector<cv::Point2d>& b, const cv::Matx33d& H,
                     cv::Size sizeB, double eps) {
    int seen = 0, found = 0;
    for (const auto& p : a) {
        const cv::Point2d q = transform(H, p);
        if (q.x < 10 || q.y < 10 || q.x > sizeB.width - 10 || q.y > sizeB.height - 10) continue;
        ++seen;
        for (const auto& r : b)
            if (cv::norm(r - q) < eps) {
                ++found;
                break;
            }
    }
    return seen ? double(found) / seen : 0.0;
}

std::vector<cv::Point2d> positions(const Features& f) {
    std::vector<cv::Point2d> out;
    for (const auto& k : f.keypoints) out.push_back(k.pt);
    return out;
}

}  // namespace

TEST(Sift, FindsABlobAtItsPositionAndScale) {
    // Scale-normalised Laplacian of a Gaussian blob of std t peaks at σ = t, at the blob's centre; the
    // centre is off-grid, so only the sub-pixel refinement can reach it.
    for (double t : {3.0, 6.0}) {
        const double cx = 80.3, cy = 71.7;
        const auto f = detectSift(blobImage({160, 144}, cx, cy, t));
        ASSERT_FALSE(f.keypoints.empty()) << "t = " << t;
        const Keypoint* best = &f.keypoints.front();
        for (const auto& k : f.keypoints)
            if (k.contrast > best->contrast) best = &k;
        EXPECT_LT(cv::norm(best->pt - cv::Point2d(cx, cy)), 0.1) << "t = " << t;
        EXPECT_NEAR(best->sigma / t, 1.0, 0.15) << "t = " << t;
    }
}

TEST(Sift, EdgeResponsesAreRejected) {
    // A thin vertical ridge (a line 3 pixels wide), slightly noisy so that it has strict extrema: the DoG
    // is strong all along it, but along it the curvature is ~0, so every response is an edge response.
    cv::Mat edge(120, 160, CV_8U);
    std::mt19937 rng(1);
    std::normal_distribution<double> noise(0, 1.0);
    for (int y = 0; y < edge.rows; ++y)
        for (int x = 0; x < edge.cols; ++x) {
            const double dx = x - 80.0;
            edge.at<uchar>(y, x) = cv::saturate_cast<uchar>(60 + 150 * std::exp(-dx * dx / (2 * 1.5 * 1.5)) + noise(rng));
        }
    SiftParams p;
    const size_t kept = detectSift(edge, p).keypoints.size();
    p.edgeRatio = 1e9;  // test disabled: the edge is full of "keypoints"
    const size_t all = detectSift(edge, p).keypoints.size();
    EXPECT_GT(all, 20u);
    EXPECT_LT(kept * 5, all) << kept << " kept of " << all;
}

TEST(Sift, RepeatableUnderAKnownHomographyLikeOpenCv) {
    // Two views of a turning camera: view 0 -> view 1 is exactly H. A repeatable detector finds the
    // same points again; OpenCV's SIFT on the same images sets the bar.
    const auto v = scenes::rotatingCamera({480, 360}, 2, 18.0, 5);
    const cv::Matx33d H = v.toReference[1].inv() * v.toReference[0];
    const auto fa = detectSift(v.images[0]), fb = detectSift(v.images[1]);
    const double ours = repeatability(positions(fa), positions(fb), H, v.images[1].size(), 2.0);
    const double theirs = repeatability(opencv::siftKeypoints(v.images[0], 3000), opencv::siftKeypoints(v.images[1], 3000), H,
                                        v.images[1].size(), 2.0);
    EXPECT_GT(fa.keypoints.size(), 300u);
    EXPECT_GT(ours, 0.5);
    EXPECT_GE(ours, 0.85 * theirs) << "ours " << ours << ", OpenCV " << theirs;
}

TEST(Sift, DescriptorsSurviveRotation) {
    // The same image turned by 90° (exact) and by 33° (resampled): the descriptor is built in the
    // keypoint's own orientation, so nearest-neighbour matches must land where the rotation sends them.
    const auto v = scenes::rotatingCamera({400, 300}, 1, 0.0, 9);
    const cv::Mat& img = v.images[0];
    const cv::Point2d centre(img.cols / 2.0, img.rows / 2.0);
    for (double deg : {90.0, 33.0}) {
        cv::Mat rotated;
        cv::Matx23d A;
        if (deg == 90.0) {
            cv::rotate(img, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);  // (x, y) -> (y, W − 1 − x)
            A = cv::Matx23d(0, 1, 0, -1, 0, img.cols - 1.0);
        } else {
            A = cv::Matx23d(cv::getRotationMatrix2D(centre, deg, 1.0));
            cv::warpAffine(img, rotated, A, img.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
        }
        const auto fa = detectSift(img), fb = detectSift(rotated);
        const auto m = matchDescriptors(fa.descriptors, fb.descriptors, {0.8f, true});
        int correct = 0;
        for (const auto& mm : m) {
            const cv::Point2d p = fa.keypoints[size_t(mm.a)].pt;
            const cv::Point2d q(A(0, 0) * p.x + A(0, 1) * p.y + A(0, 2), A(1, 0) * p.x + A(1, 1) * p.y + A(1, 2));
            correct += cv::norm(q - fb.keypoints[size_t(mm.b)].pt) < 3.0;
        }
        EXPECT_GT(correct, 150) << deg << "°";
        EXPECT_GT(double(correct) / double(m.size()), 0.9) << deg << "°: " << correct << " / " << m.size();
    }
}

TEST(Matching, EqualsANaiveReference) {
    // Random descriptors, some of b near copies of rows of a (true matches), some ambiguous pairs.
    std::mt19937 rng(3);
    std::normal_distribution<float> noise(0.f, 1.f);
    cv::Mat a(120, 128, CV_32F), b(150, 128, CV_32F);
    for (int i = 0; i < a.rows; ++i)
        for (int k = 0; k < 128; ++k) a.at<float>(i, k) = noise(rng);
    for (int j = 0; j < b.rows; ++j)
        for (int k = 0; k < 128; ++k)
            b.at<float>(j, k) = j < 100 ? a.at<float>(j % 80, k) + 0.3f * noise(rng) : noise(rng);  // rows 80..99 repeat 0..19

    for (const MatchParams p : {MatchParams{0.75f, true}, MatchParams{0.75f, false}, MatchParams{0.95f, true},
                                MatchParams{1.0f, false}}) {
        // The reference: the definitions, written the slow and obvious way.
        std::vector<std::pair<int, int>> expected;
        auto d2 = [&](int i, int j) { return cv::norm(a.row(i), b.row(j), cv::NORM_L2SQR); };
        for (int i = 0; i < a.rows; ++i) {
            int best = 0;
            for (int j = 1; j < b.rows; ++j)
                if (d2(i, j) < d2(i, best)) best = j;
            double second = 1e300;
            for (int j = 0; j < b.rows; ++j)
                if (j != best) second = std::min(second, d2(i, j));
            if (p.ratio < 1 && std::sqrt(d2(i, best) / second) >= p.ratio) continue;
            int back = 0;
            for (int k = 1; k < a.rows; ++k)
                if (d2(k, best) < d2(back, best)) back = k;
            if (p.mutual && back != i) continue;
            expected.emplace_back(i, best);
        }
        const auto got = matchDescriptors(a, b, p);
        ASSERT_EQ(got.size(), expected.size()) << "ratio " << p.ratio << " mutual " << p.mutual;
        for (size_t k = 0; k < got.size(); ++k) {
            EXPECT_EQ(got[k].a, expected[k].first);
            EXPECT_EQ(got[k].b, expected[k].second);
        }
    }
}
