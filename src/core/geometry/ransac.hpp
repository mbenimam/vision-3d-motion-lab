#pragma once
// RANSAC, random sample consensus (Fischler & Bolles 1981). Course "3D Vision and Motion", part 4.
//
// Robust fitting when part of the data is plain wrong (bad matches). The steps:
//   1. drawSample      s distinct correspondences at random (s = 4 for H, 8 for F)
//   2. fitMinimal      the model(s) through them (problem-specific)
//   3. countInliers    correspondences whose residual is below the threshold: the consensus
//   4. adaptiveTrials  keep the best model; from its inlier ratio w, update the number of trials
//                        N = log(1 − p) / log(1 − w^s)
//                      (N trials all contain an outlier with probability (1 − w^s)^N; ask it to be ≤ 1 − p)
//   5. refit           least squares on all inliers of the best model, recount, repeat while it grows
//
// The loop is the same for homographies and fundamental matrices: it is written once, as a template on
// a Problem type that provides
//     using Model = ...;
//     int size() const;                                     n correspondences
//     int sampleSize() const;                               s
//     std::vector<Model> fitMinimal(const std::vector<int>& sample) const;   0, 1 or more models
//     void residuals(const Model&, std::vector<double>& r) const;            r[i] in pixels
//     std::optional<Model> fitAll(const std::vector<int>& indices) const;    least squares
//
// Cost O(N · (fit + n)).

#include <algorithm>
#include <cstdint>
#include <optional>
#include <random>
#include <stop_token>
#include <vector>

#include "core/common/task.hpp"

namespace vl::geometry {

struct RansacParams {
    double threshold = 3.0;    // a correspondence is an inlier when its residual is below this (pixels)
    double confidence = 0.99;  // p: wanted probability of drawing at least one clean sample
    int maxTrials = 5000;      // hard cap on N
    std::uint32_t seed = 7;    // fixed, so a run is reproducible
    int refitRounds = 3;       // step 5 at most this many times
};

struct RansacReport {
    int trials = 0;                        // samples actually drawn
    int requiredTrials = 0;                // N from the final inlier ratio
    std::vector<int> bestInliersPerTrial;  // consensus of the best model so far, after each trial
    int inliersBeforeRefit = 0;
};

template <class Model>
struct RansacResult {
    std::optional<Model> model;
    std::vector<std::uint8_t> inliers;  // 1 per correspondence
    int nInliers = 0;
    RansacReport report;
};

// Step 4: N = log(1 − p) / log(1 − w^s), clamped to [1, cap]. w = 0 gives the cap.
int adaptiveTrials(double inlierRatio, int sampleSize, double confidence, int cap);

// Step 1: s distinct indices in [0, n), by rejection (cheap because s ≪ n).
std::vector<int> drawSample(std::mt19937& rng, int n, int s);

// Step 3: inlier[i] = residual[i] < threshold (a NaN residual is an outlier). Returns the count.
int countInliers(const std::vector<double>& residuals, double threshold, std::vector<std::uint8_t>& inlier);

// Indices i with mask[i] set.
std::vector<int> indicesOf(const std::vector<std::uint8_t>& mask);

template <class Problem>
RansacResult<typename Problem::Model> ransac(const Problem& problem, const RansacParams& p, std::stop_token stop = {}) {
    using Model = typename Problem::Model;
    const int n = problem.size(), s = problem.sampleSize();
    RansacResult<Model> result;
    result.inliers.assign(size_t(std::max(n, 0)), 0);
    if (n < s) return result;

    std::mt19937 rng(p.seed);
    std::vector<double> r;
    std::vector<std::uint8_t> mask(static_cast<size_t>(n));
    std::optional<Model> best;
    int bestCount = 0, needed = p.maxTrials;
    for (int trial = 0; trial < needed; ++trial) {
        if (trial % 32 == 0) checkStop(stop);
        // 1–2. A random minimal sample and the model(s) through it.
        for (const Model& model : problem.fitMinimal(drawSample(rng, n, s))) {
            // 3. Its consensus.
            problem.residuals(model, r);
            const int count = countInliers(r, p.threshold, mask);
            // 4. A better model raises the known inlier ratio, so fewer trials are needed.
            if (count > bestCount) {
                bestCount = count;
                best = model;
                needed = adaptiveTrials(double(count) / n, s, p.confidence, p.maxTrials);
            }
        }
        result.report.bestInliersPerTrial.push_back(bestCount);
        result.report.trials = trial + 1;
    }
    result.report.requiredTrials = needed;
    result.report.inliersBeforeRefit = bestCount;
    if (!best) return result;

    // 5. The minimal model saw only s points; all its inliers pin it down better. Refit, recount and
    //    repeat while the consensus grows (stop if it would shrink).
    problem.residuals(*best, r);
    bestCount = countInliers(r, p.threshold, result.inliers);
    for (int round = 0; round < p.refitRounds; ++round) {
        const std::optional<Model> refined = problem.fitAll(indicesOf(result.inliers));
        if (!refined) break;
        problem.residuals(*refined, r);
        const int count = countInliers(r, p.threshold, mask);
        if (count < bestCount) break;
        const bool grew = count > bestCount;
        best = refined;
        bestCount = count;
        result.inliers = mask;
        if (!grew) break;
    }
    result.model = best;
    result.nInliers = bestCount;
    return result;
}

}  // namespace vl::geometry
