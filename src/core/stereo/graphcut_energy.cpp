// The three terms of the Kolmogorov–Zabih stereo energy, and E(f) for a configuration.
// Course: "3D Vision and Motion", part 8. See graphcut_energy.hpp for the model.
//
//   D(a)      data: colour dissimilarity of the two pixels of a, Birchfield–Tomasi, truncated
//   K         occlusion: a constant per unmatched pixel
//   V(a1,a2)  smoothness: λ, or λ/3 across an intensity edge
//   evaluate  E(f) = dataTerm + occlusionTerm + smoothnessTerm, O(W·H)

#include "core/stereo/graphcut_energy.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace vl::stereo {
namespace {

cv::Mat colourFloat(const cv::Mat& img) {
    CV_Assert(img.depth() == CV_8U && (img.channels() == 1 || img.channels() == 3));
    cv::Mat bgr, out;
    if (img.channels() == 1)
        cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
    else
        bgr = img;
    bgr.convertTo(out, CV_32FC3);
    return out;
}

cv::Mat greyFloat(const cv::Mat& img) {
    cv::Mat grey, out;
    if (img.channels() == 3)
        cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
    else
        grey = img;
    grey.convertTo(out, CV_32F);
    return out;
}

// Birchfield–Tomasi: the image row seen as a piecewise-linear signal. Within half a pixel of x it
// takes every value between I(x) and the midpoints (I(x−1) + I(x)) / 2, (I(x) + I(x+1)) / 2; keep the
// smallest and largest of them, per channel. (Edge pixels use their own value for the missing side.)
void intervals(const cv::Mat& I, cv::Mat& lowest, cv::Mat& highest) {
    lowest.create(I.size(), CV_32FC3);
    highest.create(I.size(), CV_32FC3);
    for (int y = 0; y < I.rows; ++y)
        for (int x = 0; x < I.cols; ++x) {
            const cv::Vec3f centre = I.at<cv::Vec3f>(y, x);
            const cv::Vec3f before = 0.5f * (centre + I.at<cv::Vec3f>(y, std::max(x - 1, 0)));
            const cv::Vec3f after = 0.5f * (centre + I.at<cv::Vec3f>(y, std::min(x + 1, I.cols - 1)));
            for (int c = 0; c < 3; ++c) {
                lowest.at<cv::Vec3f>(y, x)[c] = std::min({centre[c], before[c], after[c]});
                highest.at<cv::Vec3f>(y, x)[c] = std::max({centre[c], before[c], after[c]});
            }
        }
}

// Distance from value v to the interval [lo, hi] (0 inside).
float distanceToInterval(float v, float lo, float hi) { return std::max({0.f, v - hi, lo - v}); }

}  // namespace

StereoEnergy::StereoEnergy(const cv::Mat& left, const cv::Mat& right, const GraphCutParams& p)
    : p_(p), L_(colourFloat(left)), R_(colourFloat(right)), Lgrey_(greyFloat(left)), Rgrey_(greyFloat(right)) {
    CV_Assert(left.size() == right.size() && left.type() == right.type());
    CV_Assert(p.minDisparity >= 0 && p.maxDisparity >= p.minDisparity);
    intervals(L_, Lmin_, Lmax_);
    intervals(R_, Rmin_, Rmax_);
}

// δ(p, p') averaged over the colour channels. Birchfield–Tomasi takes, per channel, the smaller of
// "distance from L(p) to the right signal near p'" and "from R(p') to the left signal near p", so a
// match that falls between two samples is not penalised for the sampling.
double StereoEnergy::dissimilarity(int x, int y, int d) const {
    const cv::Vec3f l = L_.at<cv::Vec3f>(y, x), r = R_.at<cv::Vec3f>(y, x - d);
    double sum = 0;
    for (int c = 0; c < 3; ++c) {
        if (p_.birchfieldTomasi) {
            const float leftToRight = distanceToInterval(l[c], Rmin_.at<cv::Vec3f>(y, x - d)[c], Rmax_.at<cv::Vec3f>(y, x - d)[c]);
            const float rightToLeft = distanceToInterval(r[c], Lmin_.at<cv::Vec3f>(y, x)[c], Lmax_.at<cv::Vec3f>(y, x)[c]);
            sum += std::min(leftToRight, rightToLeft);
        } else {
            sum += std::abs(l[c] - r[c]);
        }
    }
    return sum / 3;
}

// D(a) = min(δ, T) or min(δ, T)² / T. Truncation keeps one badly matched pixel (noise, a specular
// spot, an occlusion forced to match in Potts mode) from dominating its neighbourhood.
double StereoEnergy::D(int x, int y, int d) const {
    if (!exists(x, d)) return p_.truncation;  // only asked in Potts mode: the worst cost
    const double t = std::min(dissimilarity(x, y, d), p_.truncation);
    return p_.cost == DataCost::Absolute ? t : t * t / p_.truncation;
}

// V = λ, or λ/3 when the left pixels, or (with occlusions) the right pixels p0 − d, p1 − d, differ by
// at least the edge threshold: a depth edge is more likely where the image has an edge.
double StereoEnergy::V(int x0, int y0, int x1, int y1, int d) const {
    if (!p_.adaptiveLambda) return p_.lambda;
    float contrast = std::abs(Lgrey_.at<float>(y0, x0) - Lgrey_.at<float>(y1, x1));
    if (p_.modelOcclusions) contrast = std::max(contrast, std::abs(Rgrey_.at<float>(y0, x0 - d) - Rgrey_.at<float>(y1, x1 - d)));
    return contrast < p_.edgeThreshold ? p_.lambda : p_.lambda / 3;
}

EnergyTerms StereoEnergy::evaluate(const Configuration& f) const {
    return {dataTerm(f), occlusionTerm(f), smoothnessTerm(f)};
}

// E_data = Σ_{a active} D(a): one active assignment per matched left pixel.
double StereoEnergy::dataTerm(const Configuration& f) const {
    double e = 0;
    for (int y = 0; y < height(); ++y)
        for (int x = 0; x < width(); ++x) {
            const int d = f.left.at<int>(y, x);
            if (d != kOccluded) e += D(x, y, d);
        }
    return e;
}

// E_occ = K · (occluded left pixels + occluded right pixels).
double StereoEnergy::occlusionTerm(const Configuration& f) const {
    if (!p_.modelOcclusions) return 0;
    long occluded = 0;
    for (int y = 0; y < height(); ++y)
        for (int x = 0; x < width(); ++x)
            occluded += (f.left.at<int>(y, x) == kOccluded) + (f.right.at<int>(y, x) == kOccluded);
    return K() * double(occluded);
}

// E_smooth over the pairs of neighbouring left pixels p0, p1 (right and lower neighbour).
// With occlusions: of all the assignment pairs {<p0, p0 − d>, <p1, p1 − d>}, only d = d0 and d = d1 can
// have exactly one active member; if d0 = d1 both are active together (or both pixels are occluded).
// Potts: V · [d0 ≠ d1].
double StereoEnergy::smoothnessTerm(const Configuration& f) const {
    auto pair = [&](int x0, int y0, int x1, int y1) {
        const int d0 = f.left.at<int>(y0, x0), d1 = f.left.at<int>(y1, x1);
        if (d0 == d1) return 0.0;
        if (!p_.modelOcclusions) return V(x0, y0, x1, y1, 0);
        double e = 0;
        if (d0 != kOccluded && exists(x1, d0)) e += V(x0, y0, x1, y1, d0);
        if (d1 != kOccluded && exists(x0, d1)) e += V(x0, y0, x1, y1, d1);
        return e;
    };
    double e = 0;
    for (int y = 0; y < height(); ++y)
        for (int x = 0; x < width(); ++x) {
            if (x + 1 < width()) e += pair(x, y, x + 1, y);
            if (y + 1 < height()) e += pair(x, y, x, y + 1);
        }
    return e;
}

}  // namespace vl::stereo
