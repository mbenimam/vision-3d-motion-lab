#include "core/geometry/features.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "core/common/cv_compat.hpp"
#include "core/common/task.hpp"

namespace vl::geometry {

Features detectSift(const cv::Mat& image, int maxFeatures) {
    cv::Mat grey;
    if (image.channels() == 3)
        cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, grey, cv::COLOR_BGRA2GRAY);
    else
        grey = image;
    Features f;
    cv::SIFT::create(maxFeatures)->detectAndCompute(grey, cv::noArray(), f.keypoints, f.descriptors);
    if (!f.descriptors.empty() && f.descriptors.type() != CV_32F) f.descriptors.convertTo(f.descriptors, CV_32F);
    return f;
}

std::vector<Match> matchDescriptors(const cv::Mat& da, const cv::Mat& db, const MatchParams& p,
                                    std::stop_token stop) {
    std::vector<Match> out;
    const int na = da.rows, nb = db.rows;
    if (na == 0 || nb < 2) return out;
    CV_Assert(da.type() == CV_32F && db.type() == CV_32F && da.cols == db.cols);

    constexpr float kInf = std::numeric_limits<float>::infinity();
    std::vector<int> best1(size_t(na), -1);
    std::vector<float> d1(size_t(na), kInf), d2(size_t(na), kInf);

    // Per block, also the best row for each column (for the mutual check), merged afterwards.
    constexpr int kBlock = 256;
    const int nBlocks = (na + kBlock - 1) / kBlock;
    std::vector<std::vector<float>> colBest(static_cast<size_t>(nBlocks));
    std::vector<std::vector<int>> colArg(static_cast<size_t>(nBlocks));

    cv::parallel_for_(cv::Range(0, nBlocks), [&](const cv::Range& range) {
        cv::Mat dist;
        for (int blk = range.start; blk < range.end; ++blk) {
            if (stop.stop_requested()) return;
            const int r0 = blk * kBlock, r1 = std::min(na, r0 + kBlock);
            // All squared distances of this block to every descriptor of b (vectorised primitive).
            cv::batchDistance(da.rowRange(r0, r1), db, dist, CV_32F, cv::noArray(), cv::NORM_L2SQR);
            auto& cb = colBest[size_t(blk)];
            auto& ca = colArg[size_t(blk)];
            cb.assign(size_t(nb), kInf);
            ca.assign(size_t(nb), -1);
            for (int i = r0; i < r1; ++i) {
                const float* row = dist.ptr<float>(i - r0);
                float b1 = kInf, b2 = kInf;
                int arg = -1;
                for (int j = 0; j < nb; ++j) {
                    const float d = row[j];
                    if (d < b1) {
                        b2 = b1, b1 = d, arg = j;
                    } else if (d < b2) {
                        b2 = d;
                    }
                    if (d < cb[size_t(j)]) cb[size_t(j)] = d, ca[size_t(j)] = i;
                }
                best1[size_t(i)] = arg;
                d1[size_t(i)] = std::sqrt(std::max(b1, 0.f));
                d2[size_t(i)] = std::sqrt(std::max(b2, 0.f));
            }
        }
    });
    checkStop(stop);

    std::vector<int> back(size_t(nb), -1);
    if (p.mutual) {
        std::vector<float> bestD(size_t(nb), kInf);
        for (int blk = 0; blk < nBlocks; ++blk)
            for (int j = 0; j < nb; ++j)
                if (colBest[size_t(blk)][size_t(j)] < bestD[size_t(j)])
                    bestD[size_t(j)] = colBest[size_t(blk)][size_t(j)], back[size_t(j)] = colArg[size_t(blk)][size_t(j)];
    }
    out.reserve(size_t(na) / 3);
    for (int i = 0; i < na; ++i) {
        const int j = best1[size_t(i)];
        if (j < 0) continue;
        const float r = d2[size_t(i)] > 0 ? d1[size_t(i)] / d2[size_t(i)] : 1.f;
        if (r >= p.ratio && p.ratio < 1.f) continue;
        if (p.mutual && back[size_t(j)] != i) continue;
        out.push_back({i, j, d1[size_t(i)], r});
    }
    return out;
}

namespace {

// FNV-1a over the pixels (and the size): a few ms for a megapixel, far less than SIFT.
std::uint64_t imageKey(const cv::Mat& img, int maxFeatures) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&h](std::uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    mix(std::uint64_t(img.rows)), mix(std::uint64_t(img.cols)), mix(std::uint64_t(img.type())), mix(std::uint64_t(maxFeatures));
    for (int y = 0; y < img.rows; ++y) {
        const uchar* row = img.ptr<uchar>(y);  // 8 bytes at a time (memcpy: rows need not be aligned)
        const size_t bytes = img.cols * img.elemSize();
        for (size_t k = 0; k < bytes / 8; ++k) {
            std::uint64_t v;
            std::memcpy(&v, row + 8 * k, 8);
            mix(v);
        }
        const uchar* tail = row + (bytes / 8) * 8;
        for (size_t k = 0; k < bytes % 8; ++k) mix(tail[k]);
    }
    return h;
}

}  // namespace

Features FeatureCache::get(const cv::Mat& image, int maxFeatures) {
    const std::uint64_t key = imageKey(image, maxFeatures);
    {
        std::lock_guard lock(mutex_);
        for (const auto& e : entries_)
            if (e.key == key) return e.f;
    }
    Features f = detectSift(image, maxFeatures);  // outside the lock: two runs may detect in parallel
    std::lock_guard lock(mutex_);
    entries_.push_front({key, f});
    if (entries_.size() > 6) entries_.pop_back();
    return f;
}

}  // namespace vl::geometry
