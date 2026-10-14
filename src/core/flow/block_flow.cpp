// Block-matching optical flow. Course: "3D Vision and Motion", part 11.
//
// The method, one function per step:
//   1. displacementsByLength   the (2S+1)² candidates d, shortest first
//   2. costForDisplacement     C(x, d) for every pixel x at once, one d at a time: the integral
//                              image of the pixel differences, then every block mean from it
//   3. keepBest                w(x) = argmin_d C(x, d), as a running minimum over the candidates
//   4. refineSubpixel          a parabola through the costs of the neighbouring displacements,
//                              separately along x and y
//
// Cost: O(W·H·(2S+1)²) for steps 2–3 (each block mean is O(1) from the integral image), plus
// O(W·H·(2r+1)²) for step 4 (four extra block costs per pixel).

#include "core/flow/block_flow.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "core/common/filters.hpp"
#include "core/common/image.hpp"

namespace vl::flow {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// The difference of one pair of grey levels, as the chosen cost counts it.
float pixelCost(float a, float b, BlockCost cost) { return cost == BlockCost::SAD ? std::abs(a - b) : (a - b) * (a - b); }

// C(x, d) for one pixel, straight from the definition: the mean over the block pixels x' that lie
// in frame 0; frame 1 is sampled with edge replication (vl::at), as in costForDisplacement.
float blockCost(const cv::Mat& I0, const cv::Mat& I1, int x, int y, cv::Point d, int r, BlockCost cost) {
    double sum = 0;
    int count = 0;
    for (int j = std::max(0, y - r); j <= std::min(I0.rows - 1, y + r); ++j)
        for (int i = std::max(0, x - r); i <= std::min(I0.cols - 1, x + r); ++i) {
            sum += pixelCost(I0.at<float>(j, i), vl::at(I1, j + d.y, i + d.x), cost);
            ++count;
        }
    return float(sum / count);
}

// Step 1. Shortest first, so that on ties (flat areas) the smallest motion wins, the least
// surprising answer.
std::vector<cv::Point> displacementsByLength(int S) {
    std::vector<cv::Point> d;
    for (int dy = -S; dy <= S; ++dy)
        for (int dx = -S; dx <= S; ++dx) d.emplace_back(dx, dy);
    std::stable_sort(d.begin(), d.end(), [](cv::Point a, cv::Point b) { return a.dot(a) < b.dot(b); });
    return d;
}

// Step 2. C(x, d) for every pixel x, for one displacement d. The pixel costs
//     e(x) = |I0(x) − I1(x + d)|  (or its square)
// are accumulated into their integral image S(x, y) = Σ_{x' < x, y' < y} e(x', y') in one pass:
// S of a row is S of the row above plus the running sum of the current row,
//     S(x+1, y+1) = S(x+1, y) + Σ_{x' ≤ x} e(x', y).
// The sum over any block is then four corners of S. Blocks are clipped to the image.
cv::Mat costForDisplacement(const cv::Mat& I0, const cv::Mat& I1, cv::Point d, int r, BlockCost cost) {
    cv::Mat S = cv::Mat::zeros(I0.rows + 1, I0.cols + 1, CV_64F);
    for (int y = 0; y < I0.rows; ++y) {
        double rowSum = 0;
        for (int x = 0; x < I0.cols; ++x) {
            rowSum += pixelCost(I0.at<float>(y, x), vl::at(I1, y + d.y, x + d.x), cost);
            S.at<double>(y + 1, x + 1) = S.at<double>(y, x + 1) + rowSum;
        }
    }
    cv::Mat C(I0.size(), CV_32F);
    for (int y = 0; y < I0.rows; ++y)
        for (int x = 0; x < I0.cols; ++x) {
            const int x0 = std::max(0, x - r), x1 = std::min(I0.cols, x + r + 1);
            const int y0 = std::max(0, y - r), y1 = std::min(I0.rows, y + r + 1);
            const double sum = S.at<double>(y1, x1) - S.at<double>(y0, x1) - S.at<double>(y1, x0) + S.at<double>(y0, x0);
            C.at<float>(y, x) = float(sum / ((x1 - x0) * (y1 - y0)));  // the mean over the block
        }
    return C;
}

// Step 3. Keep d where it beats the best so far. A displacement that sends the pixel itself out of
// frame 1 is not a candidate.
void keepBest(const cv::Mat& C, cv::Point d, cv::Mat& bestCost, cv::Mat& bestD) {
    for (int y = std::max(0, -d.y); y < std::min(C.rows, C.rows - d.y); ++y)
        for (int x = std::max(0, -d.x); x < std::min(C.cols, C.cols - d.x); ++x)
            if (C.at<float>(y, x) < bestCost.at<float>(y, x)) {
                bestCost.at<float>(y, x) = C.at<float>(y, x);
                bestD.at<cv::Point>(y, x) = d;
            }
}

// Vertex of the parabola through (−1, c₋), (0, c₀), (+1, c₊): t = (c₋ − c₊) / (2(c₋ − 2c₀ + c₊)),
// clamped to half a pixel (beyond that the neighbour would have won).
float parabolaVertex(float cm, float c0, float cp) {
    const float curvature = cm - 2 * c0 + cp;
    if (!std::isfinite(cm) || !std::isfinite(cp) || !(curvature > 1e-6f)) return 0.f;
    return std::clamp(0.5f * (cm - cp) / curvature, -0.5f, 0.5f);
}

// Step 4. Refine each integer winner with one parabola along x and one along y.
cv::Mat refineSubpixel(const cv::Mat& I0, const cv::Mat& I1, const cv::Mat& bestD, const cv::Mat& bestCost,
                       const BlockFlowParams& p, std::stop_token stop) {
    cv::Mat flow(I0.size(), CV_32FC2);
    for (int y = 0; y < I0.rows; ++y) {
        if (y % 32 == 0) checkStop(stop);
        for (int x = 0; x < I0.cols; ++x) {
            const cv::Point d = bestD.at<cv::Point>(y, x);
            const auto cost = [&](int ex, int ey) {  // C(x, d + e), infinite outside the search or the frame
                const cv::Point e = d + cv::Point(ex, ey), target(x + e.x, y + e.y);
                const bool searched = std::abs(e.x) <= p.search && std::abs(e.y) <= p.search;
                const bool inFrame = target.x >= 0 && target.y >= 0 && target.x < I0.cols && target.y < I0.rows;
                return searched && inFrame ? blockCost(I0, I1, x, y, e, p.radius, p.cost) : kInf;
            };
            const float c0 = bestCost.at<float>(y, x);
            flow.at<cv::Vec2f>(y, x) = {float(d.x) + (p.subpixel ? parabolaVertex(cost(-1, 0), c0, cost(1, 0)) : 0.f),
                                        float(d.y) + (p.subpixel ? parabolaVertex(cost(0, -1), c0, cost(0, 1)) : 0.f)};
        }
    }
    return flow;
}

}  // namespace

BlockFlowResult blockMatchFlow(const cv::Mat& frame0, const cv::Mat& frame1, const BlockFlowParams& p,
                               std::stop_token stop, const ProgressFn& progress) {
    CV_Assert(frame0.size() == frame1.size() && p.radius >= 0 && p.search >= 0);
    const cv::Mat I0 = toGrayF(frame0), I1 = toGrayF(frame1);
    // 1. The candidates.
    const std::vector<cv::Point> displacements = displacementsByLength(p.search);

    // 2 and 3. Cost of each candidate for the whole image, and the running winner.
    BlockFlowResult res;
    res.cost = cv::Mat(I0.size(), CV_32F, cv::Scalar(kInf));
    cv::Mat bestD(I0.size(), CV_32SC2, cv::Scalar(0, 0));
    for (size_t k = 0; k < displacements.size(); ++k) {
        if (k % 16 == 0) {
            checkStop(stop);
            report(progress, 0.9 * double(k) / double(displacements.size()),
                   "displacement " + std::to_string(k + 1) + " / " + std::to_string(displacements.size()));
        }
        const cv::Mat C = costForDisplacement(I0, I1, displacements[k], p.radius, p.cost);
        keepBest(C, displacements[k], res.cost, bestD);
    }
    res.evaluations = long(I0.total()) * long(displacements.size());

    // 4. Sub-pixel refinement (or just the integer winner).
    res.flow = refineSubpixel(I0, I1, bestD, res.cost, p, stop);
    report(progress, 1.0, "done");
    return res;
}

cv::Mat blockCostSurface(const cv::Mat& I0, const cv::Mat& I1, cv::Point p, const BlockFlowParams& params) {
    CV_Assert(I0.type() == CV_32F && I1.type() == CV_32F && I0.size() == I1.size());
    const int S = params.search;
    cv::Mat out(2 * S + 1, 2 * S + 1, CV_32F);
    for (int dy = -S; dy <= S; ++dy)
        for (int dx = -S; dx <= S; ++dx) {
            const bool inFrame = p.x + dx >= 0 && p.y + dy >= 0 && p.x + dx < I0.cols && p.y + dy < I0.rows;
            out.at<float>(dy + S, dx + S) = inFrame ? blockCost(I0, I1, p.x, p.y, {dx, dy}, params.radius, params.cost) : kInf;
        }
    return out;
}

}  // namespace vl::flow
