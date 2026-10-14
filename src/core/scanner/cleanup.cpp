// Illumination flattening and adaptive thresholding: the "scan" look. Course "3D Vision and Motion":
// the scanner application (integral images are the trick of part 6's window sums).
//
//   1. estimateIllumination  shrink (block mean) → close (max filter, then min filter) → blur → enlarge
//   2. flatten               F = I / (L̂^s · L̄^(1−s))
//   3. contrastStretch       out = 255 · (F − b) / (w − b), clamped, b and w from percentiles of F
//   4. adaptiveThreshold     out = 255 if I ≥ μ_r(x, y) − C else 0, μ_r from the integral image
//
// Why closing and not just a blur: a blur of the page averages ink into the "light", so a dense
// paragraph would look like a shadow and be brightened into grey mush. The max filter first replaces
// every stroke thinner than the window by the paper beside it; the min filter then gives large dark
// regions (a real shadow's edge) back their size, so the estimate follows the shadow, not the text.

#include "core/scanner/cleanup.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "core/common/filters.hpp"

namespace vl::scanner {
namespace {

// Step 1a: average each k × k block (the last blocks may be partial). Cheap and alias-free.
cv::Mat shrink(const cv::Mat& f, int k) {
    cv::Mat sum = cv::Mat::zeros((f.rows + k - 1) / k, (f.cols + k - 1) / k, CV_32F);
    cv::Mat count = cv::Mat::zeros(sum.size(), CV_32F);
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x) {
            sum.at<float>(y / k, x / k) += f.at<float>(y, x);
            count.at<float>(y / k, x / k) += 1;
        }
    return sum / count;
}

// Step 1b: the maximum (or minimum) over a (2r+1)² square. A square is a row segment times a column
// segment, so the filter runs along rows, then along columns: 2(2r+1) comparisons per pixel.
cv::Mat rankFilter(const cv::Mat& f, int r, bool takeMax) {
    auto pick = [takeMax](float a, float b) { return takeMax ? std::max(a, b) : std::min(a, b); };
    cv::Mat rows(f.size(), CV_32F), out(f.size(), CV_32F);
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x) {
            float v = f.at<float>(y, x);
            for (int i = -r; i <= r; ++i) v = pick(v, vl::at(f, y, x + i));
            rows.at<float>(y, x) = v;
        }
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x) {
            float v = rows.at<float>(y, x);
            for (int j = -r; j <= r; ++j) v = pick(v, vl::at(rows, y + j, x));
            out.at<float>(y, x) = v;
        }
    return out;
}

// Step 1d: back to full size, each pixel centre mapped to the small grid ((x + ½) / k − ½).
cv::Mat enlarge(const cv::Mat& small, cv::Size size, int k) {
    cv::Mat out(size, CV_32F);
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x)
            out.at<float>(y, x) =
                vl::bilinear(small, (float(x) + 0.5f) / float(k) - 0.5f, (float(y) + 0.5f) / float(k) - 0.5f);
    return out;
}

// Step 2 on one channel: F = I / (L̂^s · L̄^(1−s)), where L̄ is the mean light. Paper ends up near 1.
cv::Mat flatten(const cv::Mat& I, const cv::Mat& L, double s) {
    const double meanLight = cv::mean(L)[0];
    cv::Mat F(I.size(), CV_32F);
    for (int y = 0; y < I.rows; ++y)
        for (int x = 0; x < I.cols; ++x) {
            const double light = std::pow(std::max(1.0f, L.at<float>(y, x)), s) * std::pow(meanLight, 1 - s);
            F.at<float>(y, x) = float(I.at<float>(y, x) / light);
        }
    return F;
}

// The q-quantile of the values of f (CV_32F), by selection in O(n).
double quantile(const cv::Mat& f, double q) {
    std::vector<float> v;
    v.reserve(f.total());
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x) v.push_back(f.at<float>(y, x));
    const size_t k = std::min(v.size() - 1, size_t(q * double(v.size())));
    std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(k), v.end());
    return v[k];
}

// Step 3: b = the darkest 1 % (ink), w = 0.92 × the median (paper; paper grain a little darker than
// its median still turns white). out = 255 (F − b) / (w − b), clamped to [0, 255].
cv::Mat contrastStretch(const cv::Mat& F, double b, double w) {
    cv::Mat out(F.size(), CV_8U);
    for (int y = 0; y < F.rows; ++y)
        for (int x = 0; x < F.cols; ++x)
            out.at<uchar>(y, x) = cv::saturate_cast<uchar>(255.0 * (F.at<float>(y, x) - b) / std::max(1e-3, w - b));
    return out;
}

cv::Mat toFloat(const cv::Mat& m) {
    cv::Mat f;
    m.convertTo(f, CV_32F);
    return f;
}

}  // namespace

cv::Mat estimateIllumination(const cv::Mat& I) {
    // Work on a page at most ~300 px long: the light varies slowly, so nothing is lost, and the
    // closing's window (5 px there) spans ~ 5k px of the page, wider than any pen stroke.
    const int k = std::max(1, int(std::lround(std::max(I.rows, I.cols) / 300.0)));
    const cv::Mat small = shrink(I, k);                                       // 1a
    const cv::Mat closed = rankFilter(rankFilter(small, 5, true), 5, false);  // 1b: closing
    const cv::Mat smooth = vl::gaussianBlur(closed, 3.0);                     // 1c
    return enlarge(smooth, I.size(), k);                                      // 1d
}

cv::Mat integralImage(const cv::Mat& f) {
    // One pass: S(x+1, y+1) = f(x, y) + S(x, y+1) + S(x+1, y) − S(x, y).
    cv::Mat S = cv::Mat::zeros(f.rows + 1, f.cols + 1, CV_64F);
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x)
            S.at<double>(y + 1, x + 1) =
                f.at<float>(y, x) + S.at<double>(y, x + 1) + S.at<double>(y + 1, x) - S.at<double>(y, x);
    return S;
}

cv::Mat adaptiveThreshold(const cv::Mat& grey, int r, double C) {
    const cv::Mat f = grey.type() == CV_32F ? grey : toFloat(grey);
    const cv::Mat S = integralImage(f);
    cv::Mat out(f.size(), CV_8U);
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x) {
            // The window [x0, x1) × [y0, y1), clipped to the page: its sum from four corners of S.
            const int x0 = std::max(0, x - r), x1 = std::min(f.cols, x + r + 1);
            const int y0 = std::max(0, y - r), y1 = std::min(f.rows, y + r + 1);
            const double sum =
                S.at<double>(y1, x1) - S.at<double>(y0, x1) - S.at<double>(y1, x0) + S.at<double>(y0, x0);
            const double mean = sum / double((x1 - x0) * (y1 - y0));
            out.at<uchar>(y, x) = f.at<float>(y, x) >= mean - C ? 255 : 0;
        }
    return out;
}

CleanupResult cleanUp(const cv::Mat& page, const CleanupParams& p) {
    CV_Assert(page.type() == CV_8UC3);
    std::vector<cv::Mat> channels;
    if (p.mode == ScanMode::Colour) {
        cv::split(page, channels);
    } else {
        cv::Mat g;
        cv::cvtColor(page, g, cv::COLOR_BGR2GRAY);
        channels = {g};
    }
    // Each colour channel gets its own light: a warm lamp is divided out too (a white balance).
    std::vector<cv::Mat> light, flat;
    for (const cv::Mat& ch : channels) {
        const cv::Mat I = toFloat(ch);
        light.push_back(estimateIllumination(I));                   // 1.
        flat.push_back(flatten(I, light.back(), p.shadowRemoval));  // 2.
    }
    // 3. One black point and one white point for all channels (from their mean), so colours keep their hue.
    cv::Mat meanFlat = cv::Mat::zeros(page.size(), CV_32F);
    for (const cv::Mat& F : flat) meanFlat += F / double(flat.size());
    const double b = quantile(meanFlat, 0.01), w = 0.92 * quantile(meanFlat, 0.5);
    std::vector<cv::Mat> stretched, light8;
    for (size_t i = 0; i < flat.size(); ++i) {
        stretched.push_back(contrastStretch(flat[i], b, w));
        light8.push_back(cv::Mat());
        light[i].convertTo(light8.back(), CV_8U);
    }
    CleanupResult r;
    cv::merge(light8, r.illumination);
    cv::merge(stretched, r.scan);
    if (p.mode == ScanMode::BlackWhite) r.scan = adaptiveThreshold(r.scan, p.window, p.C);  // 4.
    return r;
}

}  // namespace vl::scanner
