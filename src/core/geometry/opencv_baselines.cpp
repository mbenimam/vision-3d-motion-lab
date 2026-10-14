#include "core/geometry/opencv_baselines.hpp"

#include "core/common/cv_compat.hpp"
#include "core/common/task.hpp"

namespace vl::geometry::opencv {
namespace {

cv::Mat grey(const cv::Mat& img) {
    if (img.channels() == 1) return img;
    cv::Mat g;
    cv::cvtColor(img, g, img.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
    return g;
}

std::vector<cv::Point2f> toFloat(const Points& pts) {
    std::vector<cv::Point2f> out;
    out.reserve(pts.size());
    for (const auto& p : pts) out.emplace_back(float(p.x), float(p.y));
    return out;
}

std::vector<std::uint8_t> toMask(const cv::Mat& m, size_t n, int& count) {
    std::vector<std::uint8_t> out(n, 0);
    count = 0;
    for (size_t i = 0; i < n && !m.empty(); ++i) {
        out[i] = m.at<uchar>(int(i)) ? 1 : 0;
        count += out[i];
    }
    return out;
}

}  // namespace

Correspondences siftMatches(const cv::Mat& imageA, const cv::Mat& imageB, float ratio, int maxFeatures) {
    const Stopwatch sw;
    auto sift = cv::SIFT::create(maxFeatures);
    std::vector<cv::KeyPoint> ka, kb;
    cv::Mat da, db;
    sift->detectAndCompute(grey(imageA), cv::noArray(), ka, da);
    sift->detectAndCompute(grey(imageB), cv::noArray(), kb, db);
    Correspondences c;
    c.keypointsA = int(ka.size());
    c.keypointsB = int(kb.size());
    if (da.rows > 0 && db.rows >= 2) {
        std::vector<std::vector<cv::DMatch>> knn;
        cv::BFMatcher(cv::NORM_L2).knnMatch(da, db, knn, 2);
        for (const auto& m : knn)
            if (m.size() == 2 && m[0].distance < ratio * m[1].distance) {
                c.a.emplace_back(ka[size_t(m[0].queryIdx)].pt);
                c.b.emplace_back(kb[size_t(m[0].trainIdx)].pt);
            }
    }
    c.ms = sw.ms();
    return c;
}

std::vector<cv::Point2d> siftKeypoints(const cv::Mat& image, int maxFeatures) {
    std::vector<cv::KeyPoint> k;
    cv::SIFT::create(maxFeatures)->detect(grey(image), k);
    std::vector<cv::Point2d> out;
    for (const auto& kp : k) out.emplace_back(kp.pt);
    return out;
}

Homography findHomography(const Points& x, const Points& xp, double threshold, double confidence) {
    Homography h;
    if (x.size() < 4) return h;
    cv::Mat mask;
    const Stopwatch sw;
    const cv::Mat H = cv::findHomography(toFloat(x), toFloat(xp), cv::RANSAC, threshold, mask, 2000, confidence);
    h.ms = sw.ms();
    if (H.empty()) return h;
    h.H = cv::Matx33d(H);
    h.ok = true;
    h.inliers = toMask(mask, x.size(), h.nInliers);
    return h;
}

Fundamental findFundamental(const Points& x1, const Points& x2, double threshold, double confidence) {
    Fundamental f;
    if (x1.size() < 8) return f;
    cv::Mat mask;
    const Stopwatch sw;
    // OpenCV's convention is ours: x2ᵀ F x1 = 0 with points1 = x1. It may return several 3×3 stacked.
    const cv::Mat F = cv::findFundamentalMat(toFloat(x1), toFloat(x2), cv::FM_RANSAC, threshold, confidence, mask);
    f.ms = sw.ms();
    if (F.rows < 3) return f;
    f.F = cv::Matx33d(F(cv::Rect(0, 0, 3, 3)));
    f.F *= 1.0 / cv::norm(f.F);
    f.ok = true;
    f.inliers = toMask(mask, x1.size(), f.nInliers);
    return f;
}

}  // namespace vl::geometry::opencv
