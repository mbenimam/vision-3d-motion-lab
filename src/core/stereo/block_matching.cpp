// Local stereo by block matching. Course: "3D Vision and Motion", part 6.
//
// The method in four steps, one function each:
//   1. costForDisparity  C(x, y, d) = cost of matching the window around left (x, y) with the window
//                        around right (x − d, y), for every pixel, one disparity at a time
//   2. winnerTakesAll    d*(x, y) = argmin_d C(x, y, d)
//   3. subpixel          refine d* with the parabola through C(d*−1), C(d*), C(d*+1)
//   4. leftRightCheck    keep d* only if matching right-to-left gives the same answer
//
// Window sums come from an integral image, so each cost is O(1) per pixel whatever the window
// size, and the whole method costs O(W·H·D).

#include "core/stereo/block_matching.hpp"

#include <cmath>
#include <limits>

#include "core/common/image.hpp"

namespace vl::stereo {
namespace {

constexpr float kNoMatch = std::numeric_limits<float>::infinity();
const float kNaN = std::numeric_limits<float>::quiet_NaN();

// The integral image S(x, y) = Σ_{x' < x, y' < y} f(x', y'), one row and one column larger than f.
// One pass: S(x+1, y+1) = f(x, y) + S(x, y+1) + S(x+1, y) − S(x, y).
cv::Mat integralImage(const cv::Mat& f) {
    cv::Mat S = cv::Mat::zeros(f.rows + 1, f.cols + 1, CV_64F);
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x)
            S.at<double>(y + 1, x + 1) =
                f.at<float>(y, x) + S.at<double>(y, x + 1) + S.at<double>(y + 1, x) - S.at<double>(y, x);
    return S;
}

// The right image moved d columns to the right, so that column x holds right(x − d) and lines up
// with left(x). The first d columns have no partner and stay 0 (they are excluded later).
cv::Mat shiftedRight(const cv::Mat& right, int d) {
    cv::Mat s = cv::Mat::zeros(right.size(), right.type());
    if (d < right.cols) right.colRange(0, right.cols - d).copyTo(s.colRange(d, right.cols));
    return s;
}

// Window sums of each image, computed once and reused for every disparity.
struct WindowSums {
    cv::Mat count, sumL, sumL2, sumR, sumR2;  // CV_32F, one value per pixel
};

}  // namespace

cv::Mat boxSum(const cv::Mat& f, int r) {
    CV_Assert(f.type() == CV_32F && r >= 0);
    const cv::Mat S = integralImage(f);
    cv::Mat out(f.size(), CV_32F);
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x) {
            // The window [x0, x1) × [y0, y1), clipped to the image: four corners of S give its sum.
            const int x0 = std::max(0, x - r), x1 = std::min(f.cols, x + r + 1);
            const int y0 = std::max(0, y - r), y1 = std::min(f.rows, y + r + 1);
            out.at<float>(y, x) =
                float(S.at<double>(y1, x1) - S.at<double>(y0, x1) - S.at<double>(y1, x0) + S.at<double>(y0, x0));
        }
    return out;
}

namespace {

// Step 1, for one disparity d: the cost of every left pixel against right(x − d).
//   SAD   mean |L − R| over the window
//   SSD   mean (L − R)² over the window
//   ZNCC  1 − Σ(L − μL)(R − μR) / √(Σ(L − μL)² · Σ(R − μR)²), in [0, 2]; 0 for windows of the same shape
cv::Mat costForDisparity(const cv::Mat& L, const cv::Mat& R, const WindowSums& w, int d, int r, Cost cost) {
    const cv::Mat Rd = shiftedRight(R, d);
    cv::Mat c(L.size(), CV_32F);
    if (cost == Cost::SAD) {
        c = boxSum(cv::abs(L - Rd), r) / w.count;
    } else if (cost == Cost::SSD) {
        const cv::Mat diff = L - Rd;
        c = boxSum(diff.mul(diff), r) / w.count;
    } else {
        // Σ(L − μL)(R − μR) = ΣLR − ΣL·ΣR / n, and the variances likewise. The window sums of the
        // shifted right image are the right image's own sums, shifted by d.
        const cv::Mat sumLR = boxSum(L.mul(Rd), r);
        const cv::Mat sumR = shiftedRight(w.sumR, d), sumR2 = shiftedRight(w.sumR2, d);
        for (int y = 0; y < L.rows; ++y)
            for (int x = 0; x < L.cols; ++x) {
                const double n = w.count.at<float>(y, x);
                const double sL = w.sumL.at<float>(y, x), sR = sumR.at<float>(y, x);
                const double covariance = sumLR.at<float>(y, x) - sL * sR / n;
                const double varL = w.sumL2.at<float>(y, x) - sL * sL / n;
                const double varR = sumR2.at<float>(y, x) - sR * sR / n;
                // A flat window has no variance, hence no shape to compare: guard the division.
                c.at<float>(y, x) = float(1.0 - covariance / std::sqrt(std::max(varL, 1e-3) * std::max(varR, 1e-3)));
            }
    }
    // Pixels whose window would reach past the left edge of the right image have no partner.
    c.colRange(0, std::min(L.cols, d + r)).setTo(kNoMatch);
    return c;
}

// Step 2: for each left pixel, the disparity of least cost (−1 if no disparity has a partner).
cv::Mat winnerTakesAll(const std::vector<cv::Mat>& volume, int minDisparity) {
    const cv::Size size = volume.front().size();
    cv::Mat best(size, CV_32S, cv::Scalar(-1));
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x) {
            float bestCost = kNoMatch;
            for (int k = 0; k < int(volume.size()); ++k) {
                const float c = volume[size_t(k)].at<float>(y, x);
                if (c < bestCost) {
                    bestCost = c;
                    best.at<int>(y, x) = minDisparity + k;
                }
            }
        }
    return best;
}

// Step 3: the parabola through (−1, c₋), (0, c₀), (+1, c₊) has its minimum at
// t = (c₋ − c₊) / (2 (c₋ − 2c₀ + c₊)), a fraction of a pixel from the integer winner.
float subpixelOffset(const std::vector<cv::Mat>& volume, int k, int y, int x) {
    if (k <= 0 || k + 1 >= int(volume.size())) return 0.f;
    const float cm = volume[size_t(k - 1)].at<float>(y, x);
    const float c0 = volume[size_t(k)].at<float>(y, x);
    const float cp = volume[size_t(k + 1)].at<float>(y, x);
    const float curvature = cm - 2 * c0 + cp;
    if (!std::isfinite(cm) || !std::isfinite(cp) || curvature <= 1e-6f) return 0.f;
    return 0.5f * (cm - cp) / curvature;
}

// Step 4 needs the right-to-left answer. It comes from the same volume: right pixel xr at
// disparity d is left pixel xr + d.
cv::Mat rightToLeft(const std::vector<cv::Mat>& volume, int minDisparity) {
    const cv::Size size = volume.front().size();
    cv::Mat d(size, CV_32F, cv::Scalar(kNaN));
    for (int y = 0; y < size.height; ++y)
        for (int xr = 0; xr < size.width; ++xr) {
            float bestCost = kNoMatch;
            for (int k = 0; k < int(volume.size()); ++k) {
                const int xl = xr + minDisparity + k;
                if (xl >= size.width) break;
                const float c = volume[size_t(k)].at<float>(y, xl);
                if (c < bestCost) {
                    bestCost = c;
                    d.at<float>(y, xr) = float(minDisparity + k);
                }
            }
        }
    return d;
}

}  // namespace

BlockMatchResult blockMatch(const cv::Mat& left, const cv::Mat& right, const BlockMatchParams& p,
                            std::stop_token stop, const ProgressFn& progress) {
    CV_Assert(left.size() == right.size() && p.maxDisparity >= p.minDisparity && p.minDisparity >= 0);
    const cv::Mat L = toGrayF(left), R = toGrayF(right);
    const int r = p.radius, D = p.maxDisparity - p.minDisparity + 1;
    const WindowSums w{boxSum(cv::Mat::ones(L.size(), CV_32F), r), boxSum(L, r), boxSum(L.mul(L), r), boxSum(R, r),
                       boxSum(R.mul(R), r)};

    // 1. The cost volume, one slice per disparity.
    std::vector<cv::Mat> volume;
    for (int k = 0; k < D; ++k) {
        checkStop(stop);
        report(progress, 0.9 * k / D, "matching disparity " + std::to_string(p.minDisparity + k));
        volume.push_back(costForDisparity(L, R, w, p.minDisparity + k, r, p.cost));
    }

    // 2 and 3. Winner takes all, refined to a fraction of a pixel.
    const cv::Mat best = winnerTakesAll(volume, p.minDisparity);
    BlockMatchResult res;
    res.minDisparity = p.minDisparity;
    res.disparity.create(L.size(), CV_32F);
    for (int y = 0; y < L.rows; ++y)
        for (int x = 0; x < L.cols; ++x) {
            const int d = best.at<int>(y, x);
            const float refine = p.subpixel && d >= 0 ? subpixelOffset(volume, d - p.minDisparity, y, x) : 0.f;
            res.disparity.at<float>(y, x) = d < 0 ? kNaN : float(d) + refine;
        }

    // 4. Left-right consistency: a left pixel and the right pixel it chose must choose each other.
    checkStop(stop);
    res.rightDisparity = rightToLeft(volume, p.minDisparity);
    res.rejected = cv::Mat::zeros(L.size(), CV_8U);
    if (p.leftRightCheck)
        for (int y = 0; y < L.rows; ++y)
            for (int x = 0; x < L.cols; ++x) {
                const float d = res.disparity.at<float>(y, x);
                const int xr = int(std::lround(float(x) - d));
                const bool consistent = std::isfinite(d) && xr >= 0 && xr < L.cols &&
                                        std::abs(res.rightDisparity.at<float>(y, xr) - d) <= p.lrTolerance + 0.5f;
                if (!consistent) {
                    res.rejected.at<uchar>(y, x) = 255;
                    res.disparity.at<float>(y, x) = kNaN;
                }
            }
    if (p.keepVolume) res.volume = std::move(volume);
    report(progress, 1.0, "done");
    return res;
}

double badPixelRate(const cv::Mat& disparity, const cv::Mat& truth, const cv::Mat& mask, double threshold) {
    CV_Assert(disparity.size() == truth.size());
    long total = 0, bad = 0;
    for (int y = 0; y < truth.rows; ++y)
        for (int x = 0; x < truth.cols; ++x) {
            const float t = truth.at<float>(y, x);
            if (t <= 0 || (!mask.empty() && mask.at<uchar>(y, x) == 0)) continue;  // unknown or occluded
            ++total;
            const float d = disparity.at<float>(y, x);
            if (!std::isfinite(d) || std::abs(d - t) > threshold) ++bad;
        }
    return total ? double(bad) / double(total) : 0.0;
}

}  // namespace vl::stereo
