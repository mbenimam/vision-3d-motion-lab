// Regularisation of a foreground mask: morphological filter and Potts MRF minimised by ICM
// (see regularize.hpp). Course: "3D Vision and Motion", part 9.

#include "core/motion/regularize.hpp"

#include <algorithm>

#include "core/common/filters.hpp"
#include "core/common/task.hpp"

namespace vl::motion {

cv::Mat alternatingSequentialFilter(const cv::Mat& mask, int order) {
    CV_Assert(mask.type() == CV_8U && order >= 1);
    // Small discs first: each radius removes what is smaller than it, so the larger disc that
    // follows never has to deal with the specks (that is what makes the sequence "alternating").
    cv::Mat m = mask.clone();
    for (int r = 1; r <= order; ++r) m = vl::close(vl::open(m, r), r);
    return m;
}

cv::Mat detectionMargin(const cv::Mat& difference, const cv::Mat& threshold) {
    CV_Assert(difference.type() == CV_32F && threshold.type() == CV_32F && difference.size() == threshold.size());
    cv::Mat r(difference.size(), CV_32F);
    for (int y = 0; y < r.rows; ++y)
        for (int x = 0; x < r.cols; ++x)
            r.at<float>(y, x) = difference.at<float>(y, x) / std::max(threshold.at<float>(y, x), 1.f);
    return r;
}

namespace {

// The 8 neighbours t of a pixel s.
constexpr int kNeighbours = 8;
constexpr int kDx[kNeighbours] = {-1, 0, 1, -1, 1, -1, 0, 1};
constexpr int kDy[kNeighbours] = {-1, -1, -1, 0, 0, 1, 1, 1};

// Data term of label l at margin r: ½ (r − 2l)². Its two values cross at r = 1, the threshold.
double dataCost(float r, int l, double rMax) {
    const double v = std::min(double(r), rMax) - 2.0 * l;
    return 0.5 * v * v;
}

// Step 1: the part of U that depends on x_s, for x_s = 0 and x_s = 1:
//   U_s(l) = ½ (r_s − 2l)² + β · #{t ~ s : x_t ≠ l}
struct LocalEnergy {
    double e0 = 0, e1 = 0;
};

LocalEnergy localEnergies(const cv::Mat& margin, const cv::Mat& x, int py, int px, const MrfParams& p) {
    int ones = 0, zeros = 0;
    for (int k = 0; k < kNeighbours; ++k) {
        const int ny = py + kDy[k], nx = px + kDx[k];
        if (ny < 0 || ny >= x.rows || nx < 0 || nx >= x.cols) continue;  // no neighbour beyond the edge
        if (x.at<uchar>(ny, nx))
            ++ones;
        else
            ++zeros;
    }
    const float r = margin.at<float>(py, px);
    return {dataCost(r, 0, p.rMax) + p.beta * ones, dataCost(r, 1, p.rMax) + p.beta * zeros};
}

// Step 2: one raster sweep. Each pixel takes the label of lower local energy (ties keep the current
// label, so ICM cannot cycle). Only the pixel's data term and its 8 pair terms change, so U moves by
// exactly e_new − e_old: the energy is kept up to date without recomputing it. Returns the moves.
long icmSweep(const cv::Mat& margin, cv::Mat& x, const MrfParams& p, double& U) {
    long moves = 0;
    for (int y = 0; y < x.rows; ++y)
        for (int px = 0; px < x.cols; ++px) {
            const LocalEnergy e = localEnergies(margin, x, y, px, p);
            const uchar current = x.at<uchar>(y, px);
            const uchar best = e.e1 < e.e0 ? 1 : (e.e0 < e.e1 ? 0 : current);
            if (best == current) continue;
            U += best ? e.e1 - e.e0 : e.e0 - e.e1;
            x.at<uchar>(y, px) = best;
            ++moves;
        }
    return moves;
}

}  // namespace

double mrfEnergy(const cv::Mat& margin, const cv::Mat& labels, const MrfParams& p) {
    CV_Assert(margin.type() == CV_32F && labels.type() == CV_8U && margin.size() == labels.size());
    double data = 0;
    long disagreeing = 0;
    for (int y = 0; y < labels.rows; ++y)
        for (int x = 0; x < labels.cols; ++x) {
            const bool l = labels.at<uchar>(y, x) != 0;
            data += dataCost(margin.at<float>(y, x), l, p.rMax);
            // Each unordered pair once: the neighbours after s in raster order (k = 4..7).
            for (int k = 4; k < kNeighbours; ++k) {
                const int ny = y + kDy[k], nx = x + kDx[k];
                if (ny < labels.rows && nx >= 0 && nx < labels.cols) disagreeing += l != (labels.at<uchar>(ny, nx) != 0);
            }
        }
    return data + p.beta * double(disagreeing);
}

MrfResult icmPotts(const cv::Mat& margin, const cv::Mat& init, const MrfParams& p, std::stop_token stop) {
    CV_Assert(margin.type() == CV_32F && init.type() == CV_8U && margin.size() == init.size() && p.beta >= 0);
    // Labels as 0/1.
    cv::Mat x(init.size(), CV_8U);
    for (int y = 0; y < x.rows; ++y)
        for (int px = 0; px < x.cols; ++px) x.at<uchar>(y, px) = init.at<uchar>(y, px) != 0;

    MrfResult res;
    double U = mrfEnergy(margin, x, p);
    res.energy.push_back(U);
    // 3. Sweep until nothing moves.
    for (int sweep = 0; sweep < p.maxSweeps; ++sweep) {
        checkStop(stop);
        const long moves = icmSweep(margin, x, p, U);
        res.energy.push_back(U);
        res.sweeps = sweep + 1;
        if (moves == 0) break;
    }

    res.mask.create(x.size(), CV_8U);
    for (int y = 0; y < x.rows; ++y)
        for (int px = 0; px < x.cols; ++px) {
            res.mask.at<uchar>(y, px) = x.at<uchar>(y, px) ? 255 : 0;
            res.changes += (x.at<uchar>(y, px) != 0) != (init.at<uchar>(y, px) != 0);
        }
    return res;
}

}  // namespace vl::motion
