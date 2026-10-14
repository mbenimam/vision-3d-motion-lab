// The Hough transform for lines (Hough 1962, in the (θ, ρ) form of Duda & Hart 1972), used by the
// document scanner to find the page's four sides. See hough.hpp for the parametrisation.
//
//   1. accumulateVotes  A(θ, ρ) += 1 for ρ = x cos θ + y sin θ, θ within ± w of the pixel's gradient angle
//   2. findPeaks        bins that beat their 3×3 neighbourhood, strongest first, each one suppressing
//                       the weaker bins within (± suppressDeg, ± suppressRho) of it
//
// Why vote only near the gradient direction: the gradient is the normal of the edge through the
// pixel, so the pixel can only belong to lines of about that θ. Blind voting over all 180° would
// spend 7 times more (180 bins instead of 25) and let the corners of letters vote for every line through them.

#include "core/scanner/hough.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace vl::scanner {
namespace {

constexpr int kThetaBins = 180;  // one bin per degree
constexpr double kDegree = std::numbers::pi / 180.0;

// Step 1. The gradient angle is in (−180°, 180°]; a dark-to-light and a light-to-dark edge along the
// same line have opposite gradients but the same line, so θ is taken modulo 180° and ρ computed
// with that θ.
void accumulateVotes(const cv::Mat& edges, const cv::Mat& dx, const cv::Mat& dy, int window, HoughResult& h) {
    std::vector<double> cosTheta(kThetaBins), sinTheta(kThetaBins);  // computed once, not once per vote
    for (int t = 0; t < kThetaBins; ++t) {
        cosTheta[size_t(t)] = std::cos(t * kDegree);
        sinTheta[size_t(t)] = std::sin(t * kDegree);
    }
    for (int y = 0; y < edges.rows; ++y)
        for (int x = 0; x < edges.cols; ++x) {
            if (edges.at<uchar>(y, x) == 0) continue;
            const double gradientDeg = std::atan2(dy.at<float>(y, x), dx.at<float>(y, x)) / kDegree;
            const int centre = int(std::lround(gradientDeg));
            for (int t = centre - window; t <= centre + window; ++t) {
                const int theta = ((t % kThetaBins) + kThetaBins) % kThetaBins;
                const double rho = x * cosTheta[size_t(theta)] + y * sinTheta[size_t(theta)];
                h.votes.at<int>(int(std::lround(rho)) + h.rhoOffset, theta) += 1;
            }
        }
}

// A neighbour of bin (theta, row) in θ, crossing the 0°/180° seam onto the mirrored ρ row.
int neighbourVotes(const HoughResult& h, int theta, int row) {
    int r = row;
    if (theta < 0 || theta >= kThetaBins) {
        theta = (theta + kThetaBins) % kThetaBins;
        r = 2 * h.rhoOffset - row;  // ρ → −ρ
    }
    if (r < 0 || r >= h.votes.rows) return 0;
    return h.votes.at<int>(r, theta);
}

// Distance between two lines in (θ, ρ), taking the seam into account (θ in degrees).
bool closeLines(const HoughLine& a, const HoughLine& b, int suppressDeg, double suppressRho) {
    const double dTheta = std::abs(a.theta - b.theta) / kDegree;
    if (dTheta <= suppressDeg) return std::abs(a.rho - b.rho) <= suppressRho;
    if (180.0 - dTheta <= suppressDeg) return std::abs(a.rho + b.rho) <= suppressRho;
    return false;
}

// Step 2.
std::vector<HoughLine> findPeaks(const HoughResult& h, const HoughParams& p) {
    std::vector<HoughLine> candidates;
    for (int row = 0; row < h.votes.rows; ++row)
        for (int theta = 0; theta < kThetaBins; ++theta) {
            const int v = h.votes.at<int>(row, theta);
            if (v < p.minVotes) continue;
            bool isMax = true;
            for (int j = -1; j <= 1 && isMax; ++j)
                for (int i = -1; i <= 1 && isMax; ++i)
                    if ((i != 0 || j != 0) && neighbourVotes(h, theta + i, row + j) > v) isMax = false;
            if (isMax) candidates.push_back({theta * kDegree, double(row - h.rhoOffset), v});
        }
    std::sort(candidates.begin(), candidates.end(),
              [](const HoughLine& a, const HoughLine& b) { return a.votes > b.votes; });
    std::vector<HoughLine> kept;
    for (const HoughLine& c : candidates) {
        bool suppressed = false;
        for (const HoughLine& k : kept) suppressed = suppressed || closeLines(c, k, p.suppressDeg, p.suppressRho);
        if (!suppressed) kept.push_back(c);
        if (int(kept.size()) == p.maxLines) break;
    }
    return kept;
}

}  // namespace

HoughResult houghLines(const cv::Mat& edges, const cv::Mat& dx, const cv::Mat& dy, const HoughParams& p) {
    CV_Assert(edges.type() == CV_8U && dx.type() == CV_32F && dy.type() == CV_32F);
    HoughResult h;
    h.rhoOffset = int(std::ceil(std::hypot(edges.cols, edges.rows)));
    h.votes = cv::Mat::zeros(2 * h.rhoOffset + 1, kThetaBins, CV_32S);
    accumulateVotes(edges, dx, dy, p.thetaWindowDeg, h);  // 1.
    h.lines = findPeaks(h, p);                            // 2.
    return h;
}

bool intersect(const HoughLine& a, const HoughLine& b, cv::Point2d& out) {
    // Cramer's rule; the determinant is sin(θ2 − θ1), zero for parallel lines.
    const double ca = std::cos(a.theta), sa = std::sin(a.theta), cb = std::cos(b.theta), sb = std::sin(b.theta);
    const double det = ca * sb - sa * cb;
    if (std::abs(det) < 1e-6) return false;
    out = {(a.rho * sb - b.rho * sa) / det, (ca * b.rho - cb * a.rho) / det};
    return true;
}

}  // namespace vl::scanner
