#pragma once
// RANSAC (Fischler & Bolles 1981): robust fitting when part of the data is plain wrong.
//
//   repeat N times:  draw a minimal sample (s points) -> fit a model -> count the points it explains
//   keep the model with the largest consensus, then refit it by least squares on its inliers.
//
// N is adaptive: if a fraction w of the data are inliers, a sample is clean with probability w^s, and
// N samples all fail with probability (1 - w^s)^N. Asking that to be below 1 - p gives
//
//   N = log(1 - p) / log(1 - w^s),
//
// recomputed each time a better model raises the lower bound on w. The loop is generic: the caller
// supplies the minimal solver, the residuals and the refit, so the same code fits homographies and
// fundamental matrices. Cost: O(N · (fit + n)).

#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <span>
#include <stop_token>
#include <vector>

#include "core/common/task.hpp"

namespace vl::geometry {

struct RansacParams {
    double threshold = 3.0;     // a point is an inlier when its residual is below this (pixels)
    double confidence = 0.99;   // p: wanted probability of drawing at least one clean sample
    int maxTrials = 5000;       // hard cap on N
    std::uint32_t seed = 7;     // fixed, so a run is reproducible
    int refitRounds = 3;        // least-squares refit -> recount, while the consensus grows
};

struct RansacReport {
    int trials = 0;                       // samples actually drawn
    int requiredTrials = 0;               // N from the final inlier ratio
    std::vector<int> bestInliersPerTrial; // consensus of the best model so far, after each trial (sparkline)
    int inliersBeforeRefit = 0;
};

template <class Model>
struct RansacResult {
    std::optional<Model> model;
    std::vector<std::uint8_t> inliers;  // 1 per input correspondence
    int nInliers = 0;
    RansacReport report;
};

// N = log(1 - p) / log(1 - w^s), clamped to [1, cap]. w = 0 gives the cap.
int adaptiveTrials(double inlierRatio, int sampleSize, double confidence, int cap);

template <class Model>
struct RansacProblem {
    int size = 0;        // n correspondences
    int sampleSize = 4;  // s, minimal number of correspondences to fit a model
    // Minimal solver on s indices; may return 0 (degenerate sample) or several models (e.g. 7-point).
    std::function<std::vector<Model>(std::span<const int>)> fit;
    // Residual of every correspondence for a model, in pixels (out has size n).
    std::function<void(const Model&, std::vector<double>&)> residuals;
    // Least-squares fit on all given indices; nullopt when it fails.
    std::function<std::optional<Model>(std::span<const int>)> refit;
};

template <class Model>
RansacResult<Model> ransac(const RansacProblem<Model>& pb, const RansacParams& p, std::stop_token stop = {}) {
    RansacResult<Model> res;
    const int n = pb.size, s = pb.sampleSize;
    res.inliers.assign(static_cast<size_t>(std::max(n, 0)), 0);
    if (n < s || s <= 0) return res;

    std::mt19937 rng(p.seed);
    std::uniform_int_distribution<int> pick(0, n - 1);
    std::vector<int> sample(static_cast<size_t>(s));
    std::vector<double> r(static_cast<size_t>(n));

    auto count = [&](const Model& m, std::vector<std::uint8_t>* mask) {
        pb.residuals(m, r);
        int c = 0;
        for (int i = 0; i < n; ++i) {
            const bool in = r[size_t(i)] < p.threshold;  // NaN residuals are outliers
            c += in;
            if (mask) (*mask)[size_t(i)] = in;
        }
        return c;
    };

    int best = 0, needed = p.maxTrials;
    std::optional<Model> bestModel;
    for (int trial = 0; trial < needed && trial < p.maxTrials; ++trial) {
        if ((trial & 31) == 0) checkStop(stop);
        // s distinct indices by rejection: cheap because s << n.
        for (int k = 0; k < s;) {
            const int i = pick(rng);
            bool dup = false;
            for (int j = 0; j < k; ++j) dup |= sample[size_t(j)] == i;
            if (!dup) sample[size_t(k++)] = i;
        }
        for (const Model& m : pb.fit(sample)) {
            const int c = count(m, nullptr);
            if (c > best) {
                best = c;
                bestModel = m;
                needed = adaptiveTrials(double(best) / n, s, p.confidence, p.maxTrials);
            }
        }
        res.report.bestInliersPerTrial.push_back(best);
        res.report.trials = trial + 1;
    }
    res.report.requiredTrials = needed;
    res.report.inliersBeforeRefit = best;
    if (!bestModel) return res;

    // Least-squares polish: the minimal model only saw s points; all inliers pin it down better.
    // Refit, recount, and repeat while the consensus does not shrink.
    std::vector<std::uint8_t> mask(static_cast<size_t>(n));
    best = count(*bestModel, &mask);
    for (int round = 0; round < p.refitRounds && pb.refit; ++round) {
        std::vector<int> idx;
        idx.reserve(static_cast<size_t>(best));
        for (int i = 0; i < n; ++i)
            if (mask[size_t(i)]) idx.push_back(i);
        const auto refined = pb.refit(idx);
        if (!refined) break;
        std::vector<std::uint8_t> m2(static_cast<size_t>(n));
        const int c = count(*refined, &m2);
        if (c < best) break;
        const bool grew = c > best;
        bestModel = refined;
        best = c;
        mask.swap(m2);
        if (!grew) break;
    }
    res.model = bestModel;
    res.inliers = std::move(mask);
    res.nInliers = best;
    return res;
}

}  // namespace vl::geometry
