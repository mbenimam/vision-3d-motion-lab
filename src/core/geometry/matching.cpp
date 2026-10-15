// Brute-force descriptor matching with Lowe's ratio test and a mutual check. Course "3D Vision and
// Motion", part 3. The steps are listed in matching.hpp; cost O(n_a · n_b · 128).

#include "core/geometry/matching.hpp"

#include <cmath>
#include <limits>

#include "core/common/task.hpp"

namespace vl::geometry {
namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// ‖a − b‖² of two descriptor rows.
float squaredDistance(const cv::Mat& da, int i, const cv::Mat& db, int j) {
    float sum = 0;
    for (int k = 0; k < da.cols; ++k) {
        const float diff = da.at<float>(i, k) - db.at<float>(j, k);
        sum += diff * diff;
    }
    return sum;
}

// Step 1. Rows are independent, so they are spread over the cores with cv::parallel_for_ (each
// thread writes its own rows of d²: no locking). This is the only expensive step.
cv::Mat distanceMatrix(const cv::Mat& da, const cv::Mat& db, std::stop_token stop) {
    cv::Mat d2(da.rows, db.rows, CV_32F);
    cv::parallel_for_(cv::Range(0, da.rows), [&](const cv::Range& rows) {
        for (int i = rows.start; i < rows.end; ++i) {
            if (stop.stop_requested()) return;
            for (int j = 0; j < db.rows; ++j) d2.at<float>(i, j) = squaredDistance(da, i, db, j);
        }
    });
    checkStop(stop);
    return d2;
}

struct TwoNearest {
    int best = -1;
    float d1 = kInf, d2 = kInf;  // squared distances to the nearest and the second nearest
};

// Step 2, for row i.
TwoNearest twoNearest(const cv::Mat& d2, int i) {
    TwoNearest t;
    for (int j = 0; j < d2.cols; ++j) {
        const float d = d2.at<float>(i, j);
        if (d < t.d1) {
            t.d2 = t.d1;
            t.d1 = d;
            t.best = j;
        } else if (d < t.d2) {
            t.d2 = d;
        }
    }
    return t;
}

// For the mutual check: the nearest row of column j.
int nearestRow(const cv::Mat& d2, int j) {
    int best = -1;
    float bestD = kInf;
    for (int i = 0; i < d2.rows; ++i)
        if (d2.at<float>(i, j) < bestD) {
            bestD = d2.at<float>(i, j);
            best = i;
        }
    return best;
}

}  // namespace

std::vector<Match> matchDescriptors(const cv::Mat& da, const cv::Mat& db, const MatchParams& p,
                                    std::stop_token stop) {
    std::vector<Match> out;
    if (da.rows == 0 || db.rows < 2) return out;
    CV_Assert(da.type() == CV_32F && db.type() == CV_32F && da.cols == db.cols);

    // 1. All squared distances.
    const cv::Mat d2 = distanceMatrix(da, db, stop);
    for (int i = 0; i < da.rows; ++i) {
        // 2. The two nearest neighbours of a_i.
        const TwoNearest t = twoNearest(d2, i);
        const float d1 = std::sqrt(t.d1), d2nd = std::sqrt(t.d2);
        const float ratio = d2nd > 0 ? d1 / d2nd : 1.f;
        // 3. Ratio test: the nearest must be clearly nearer than the runner-up.
        if (p.ratio < 1.f && ratio >= p.ratio) continue;
        // 4. Mutual check: a_i must be b_j's nearest too.
        if (p.mutual && nearestRow(d2, t.best) != i) continue;
        out.push_back({i, t.best, d1, ratio});
    }
    return out;
}

}  // namespace vl::geometry
