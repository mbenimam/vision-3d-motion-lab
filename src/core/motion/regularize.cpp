#include "core/motion/regularize.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>

#include "core/common/task.hpp"

namespace vl::motion {

cv::Mat alternatingSequentialFilter(const cv::Mat& mask, int order) {
    CV_Assert(mask.type() == CV_8U && order >= 1);
    cv::Mat m = mask.clone();
    for (int r = 1; r <= order; ++r) {
        const cv::Mat disc = cv::getStructuringElement(cv::MORPH_ELLIPSE, {2 * r + 1, 2 * r + 1});
        cv::morphologyEx(m, m, cv::MORPH_OPEN, disc);
        cv::morphologyEx(m, m, cv::MORPH_CLOSE, disc);
    }
    return m;
}

cv::Mat detectionMargin(const cv::Mat& difference, const cv::Mat& threshold) {
    CV_Assert(difference.type() == CV_32F && threshold.type() == CV_32F && difference.size() == threshold.size());
    cv::Mat r(difference.size(), CV_32F);
    for (int y = 0; y < r.rows; ++y) {
        const float* d = difference.ptr<float>(y);
        const float* t = threshold.ptr<float>(y);
        float* o = r.ptr<float>(y);
        for (int x = 0; x < r.cols; ++x) o[x] = d[x] / std::max(t[x], 1.f);
    }
    return r;
}

namespace {

// Data cost of label l at margin r: ½ (r − 2l)².
inline double dataCost(float r, int l, float rMax) {
    const double v = double(std::min(r, rMax)) - 2.0 * l;
    return 0.5 * v * v;
}

// Labels as 0/1 bytes with a one-pixel border of "no neighbour" (2), so the inner loop has no tests.
cv::Mat padded01(const cv::Mat& mask) {
    cv::Mat out(mask.rows + 2, mask.cols + 2, CV_8U, cv::Scalar(2));
    for (int y = 0; y < mask.rows; ++y) {
        const uchar* s = mask.ptr<uchar>(y);
        uchar* o = out.ptr<uchar>(y + 1) + 1;
        for (int x = 0; x < mask.cols; ++x) o[x] = s[x] != 0;
    }
    return out;
}

}  // namespace

double mrfEnergy(const cv::Mat& margin, const cv::Mat& labels, const MrfParams& p) {
    CV_Assert(margin.type() == CV_32F && labels.type() == CV_8U && margin.size() == labels.size());
    const float rMax = float(p.rMax);
    double data = 0;
    long disagree = 0;
    for (int y = 0; y < labels.rows; ++y) {
        const float* r = margin.ptr<float>(y);
        const uchar* l = labels.ptr<uchar>(y);
        const uchar* below = y + 1 < labels.rows ? labels.ptr<uchar>(y + 1) : nullptr;
        for (int x = 0; x < labels.cols; ++x) {
            const bool a = l[x] != 0;
            data += dataCost(r[x], a, rMax);
            // Each unordered pair once: right, and the three neighbours of the next row.
            if (x + 1 < labels.cols) disagree += a != (l[x + 1] != 0);
            if (below) {
                disagree += a != (below[x] != 0);
                if (x > 0) disagree += a != (below[x - 1] != 0);
                if (x + 1 < labels.cols) disagree += a != (below[x + 1] != 0);
            }
        }
    }
    return data + p.beta * double(disagree);
}

MrfResult icmPotts(const cv::Mat& margin, const cv::Mat& init, const MrfParams& p, std::stop_token stop) {
    CV_Assert(margin.type() == CV_32F && init.type() == CV_8U && margin.size() == init.size() && p.beta >= 0);
    const int H = init.rows, W = init.cols;
    const float rMax = float(p.rMax);
    cv::Mat L = padded01(init);                      // (H+2)×(W+2), 0/1 inside, 2 on the border
    cv::Mat dirty(H + 2, W + 2, CV_8U, cv::Scalar(1));  // pixels whose neighbourhood changed
    std::vector<uchar> rowDirty(size_t(H + 2), 1);
    MrfResult res;
    double U = mrfEnergy(margin, init, p);
    res.energy.push_back(U);

    for (int sweep = 0; sweep < p.maxSweeps; ++sweep) {
        checkStop(stop);
        long moved = 0;
        std::vector<uchar> nextRows(size_t(H + 2), 0);
        for (int y = 0; y < H; ++y) {
            if (!rowDirty[size_t(y + 1)]) continue;
            const float* r = margin.ptr<float>(y);
            const uchar* up = L.ptr<uchar>(y);
            uchar* mid = L.ptr<uchar>(y + 1);
            const uchar* dn = L.ptr<uchar>(y + 2);
            uchar* dirtyRow = dirty.ptr<uchar>(y + 1);
            for (int x = 0; x < W; ++x) {
                const int c = x + 1;
                if (!dirtyRow[c]) continue;
                dirtyRow[c] = 0;
                // Neighbours labelled 1 and 0 (border cells count for neither).
                int ones = 0, zeros = 0;
                for (const uchar v : {up[c - 1], up[c], up[c + 1], mid[c - 1], mid[c + 1], dn[c - 1], dn[c], dn[c + 1]}) {
                    ones += v == 1;
                    zeros += v == 0;
                }
                const double e0 = dataCost(r[x], 0, rMax) + p.beta * ones;
                const double e1 = dataCost(r[x], 1, rMax) + p.beta * zeros;
                // Ties keep the current label, so a sweep can never cycle.
                const uchar cur = mid[c];
                const uchar best = e1 < e0 ? 1 : (e0 < e1 ? 0 : cur);
                if (best == cur) continue;
                // Only this pixel's data term and its 8 pair terms change: U moves by e_best − e_cur.
                U += best ? e1 - e0 : e0 - e1;
                mid[c] = best;
                ++moved;
                for (int dy = 0; dy < 3; ++dy) {
                    uchar* d = dirty.ptr<uchar>(y + dy);
                    d[c - 1] = d[c] = d[c + 1] = 1;
                    nextRows[size_t(y + dy)] = 1;
                }
            }
        }
        rowDirty.swap(nextRows);
        res.energy.push_back(U);
        res.sweeps = sweep + 1;
        if (moved == 0) break;
    }
    res.mask = L(cv::Rect(1, 1, W, H)) * 255;
    cv::Mat diff;
    cv::compare(res.mask, cv::Mat(init != 0), diff, cv::CMP_NE);
    res.changes = cv::countNonZero(diff);
    return res;
}

}  // namespace vl::motion
