#include "core/stereo/block_matching.hpp"

#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

#include "core/common/image.hpp"

namespace vl::stereo {

cv::Mat boxSum(const cv::Mat& f, int r) {
    CV_Assert(f.type() == CV_32F && r >= 0);
    cv::Mat ii;  // (H+1)x(W+1), ii(y, x) = sum of f over [0, y) x [0, x)
    cv::integral(f, ii, CV_64F);
    cv::Mat out(f.size(), CV_32F);
    const int H = f.rows, W = f.cols;
    for (int y = 0; y < H; ++y) {
        const int y0 = std::max(0, y - r), y1 = std::min(H, y + r + 1);
        const double* top = ii.ptr<double>(y0);
        const double* bot = ii.ptr<double>(y1);
        float* o = out.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            const int x0 = std::max(0, x - r), x1 = std::min(W, x + r + 1);
            o[x] = static_cast<float>(bot[x1] - bot[x0] - top[x1] + top[x0]);
        }
    }
    return out;
}

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Right image shifted so that column x holds right(x - d); columns x < d have no partner.
cv::Mat shiftRight(const cv::Mat& r, int d) {
    cv::Mat s(r.size(), r.type(), cv::Scalar(0));
    if (d < r.cols) r.colRange(0, r.cols - d).copyTo(s.colRange(d, r.cols));
    return s;
}

}  // namespace

BlockMatchResult blockMatch(const cv::Mat& left, const cv::Mat& right, const BlockMatchParams& p,
                            std::stop_token stop, const ProgressFn& progress) {
    CV_Assert(left.size() == right.size() && p.maxDisparity >= p.minDisparity && p.minDisparity >= 0);
    const cv::Mat L = toGrayF(left), R = toGrayF(right);
    const int H = L.rows, W = L.cols, r = p.radius;
    const int D = p.maxDisparity - p.minDisparity + 1;

    // Window statistics of the left image, needed by ZNCC (mean and spread of each window).
    const cv::Mat n = boxSum(cv::Mat::ones(L.size(), CV_32F), r);
    cv::Mat sumL, sumL2, sumR, sumR2;
    if (p.cost == Cost::ZNCC) {
        sumL = boxSum(L, r);
        sumL2 = boxSum(L.mul(L), r);
        sumR = boxSum(R, r);
        sumR2 = boxSum(R.mul(R), r);
    }

    std::vector<cv::Mat> volume(static_cast<size_t>(D));
    for (int k = 0; k < D; ++k) {
        checkStop(stop);
        report(progress, 0.9 * k / D, "matching disparity " + std::to_string(p.minDisparity + k));
        const int d = p.minDisparity + k;
        const cv::Mat Rs = shiftRight(R, d);
        cv::Mat c;
        if (p.cost == Cost::SAD) {
            c = boxSum(cv::abs(L - Rs), r) / n;
        } else if (p.cost == Cost::SSD) {
            const cv::Mat diff = L - Rs;
            c = boxSum(diff.mul(diff), r) / n;
        } else {
            // ZNCC = cov(L, R) / (σL σR), cost = 1 - ZNCC in [0, 2]. Window stats of the shifted right image
            // are the right image's stats shifted by d.
            const cv::Mat sR = shiftRight(sumR, d), sR2 = shiftRight(sumR2, d);
            const cv::Mat sLR = boxSum(L.mul(Rs), r);
            c.create(L.size(), CV_32F);
            for (int y = 0; y < H; ++y) {
                const float *pn = n.ptr<float>(y), *pl = sumL.ptr<float>(y), *pl2 = sumL2.ptr<float>(y);
                const float *pr = sR.ptr<float>(y), *pr2 = sR2.ptr<float>(y), *plr = sLR.ptr<float>(y);
                float* pc = c.ptr<float>(y);
                for (int x = 0; x < W; ++x) {
                    const double cnt = pn[x];
                    const double cov = pl[x] * 1.0 * pr[x] / cnt;
                    const double vl = pl2[x] - pl[x] * 1.0 * pl[x] / cnt, vr = pr2[x] - pr[x] * 1.0 * pr[x] / cnt;
                    const double den = std::sqrt(std::max(vl, 1e-3) * std::max(vr, 1e-3));
                    pc[x] = static_cast<float>(1.0 - (plr[x] - cov) / den);
                }
            }
        }
        // No partner when the window would leave the right image.
        if (d + r > 0) c.colRange(0, std::min(W, d + r)).setTo(kInf);
        volume[static_cast<size_t>(k)] = c;
    }

    BlockMatchResult res;
    res.minDisparity = p.minDisparity;
    res.disparity.create(L.size(), CV_32F);
    res.rightDisparity.create(L.size(), CV_32F);
    res.rejected = cv::Mat::zeros(L.size(), CV_8U);
    cv::Mat bestK(L.size(), CV_32S);

    // Winner takes all, in the left image, with a parabola for sub-pixel precision.
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float best = kInf;
            int bk = -1;
            for (int k = 0; k < D; ++k) {
                const float v = volume[static_cast<size_t>(k)].at<float>(y, x);
                if (v < best) {
                    best = v;
                    bk = k;
                }
            }
            bestK.at<int>(y, x) = bk;
            float d = bk < 0 ? std::numeric_limits<float>::quiet_NaN() : float(p.minDisparity + bk);
            if (p.subpixel && bk > 0 && bk < D - 1) {
                const float c0 = volume[size_t(bk - 1)].at<float>(y, x), c2 = volume[size_t(bk + 1)].at<float>(y, x);
                const float den = c0 - 2 * best + c2;
                if (std::isfinite(c0) && std::isfinite(c2) && den > 1e-6f) d += 0.5f * (c0 - c2) / den;
            }
            res.disparity.at<float>(y, x) = d;
        }
    }
    checkStop(stop);
    // Right-to-left winner from the same volume: right pixel xr at disparity d is left pixel xr + d.
    for (int y = 0; y < H; ++y)
        for (int xr = 0; xr < W; ++xr) {
            float best = kInf;
            int bk = -1;
            for (int k = 0; k < D; ++k) {
                const int xl = xr + p.minDisparity + k;
                if (xl >= W) break;
                const float v = volume[static_cast<size_t>(k)].at<float>(y, xl);
                if (v < best) {
                    best = v;
                    bk = k;
                }
            }
            res.rightDisparity.at<float>(y, xr) = bk < 0 ? std::numeric_limits<float>::quiet_NaN() : float(p.minDisparity + bk);
        }
    if (p.leftRightCheck) {
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const float d = res.disparity.at<float>(y, x);
                const int xr = int(std::lround(float(x) - d));
                const bool ok = std::isfinite(d) && xr >= 0 && xr < W &&
                                std::abs(res.rightDisparity.at<float>(y, xr) - d) <= p.lrTolerance + 0.5f;
                if (!ok) {
                    res.rejected.at<uchar>(y, x) = 255;
                    res.disparity.at<float>(y, x) = std::numeric_limits<float>::quiet_NaN();
                }
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
            if (t <= 0 || (!mask.empty() && mask.at<uchar>(y, x) == 0)) continue;
            ++total;
            const float d = disparity.at<float>(y, x);
            if (!std::isfinite(d) || std::abs(d - t) > threshold) ++bad;
        }
    return total ? double(bad) / double(total) : 0.0;
}

}  // namespace vl::stereo
