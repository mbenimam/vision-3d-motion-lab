#pragma once
// Dense Lucas–Kanade (1981), a local optical-flow method. Course: "3D Vision and Motion", part 11.
//
// One constraint Ix·u + Iy·v + It = 0 per pixel cannot fix two unknowns, so assume the flow is
// constant over a Gaussian window G_σ around each pixel and solve the 2×2 least-squares system
//
//     [ G*Ix²   G*IxIy ] [u]     [ G*IxIt ]
//     [ G*IxIy  G*Iy²  ] [v] = − [ G*IyIt ]          J·w = −b,  J the structure tensor,
//
// in closed form at every pixel. The eigenvalues λ1 ≥ λ2 of J say how well the window pins the motion
// down: both large on texture or a corner; λ2 ≈ 0 along a straight edge (only the motion across the
// edge is measurable: the aperture problem); both ≈ 0 on a flat area. λ2 is returned as a confidence.
//
// Linearisation holds for about a pixel of motion, hence warping iterations (Gauss–Newton) on a
// Gaussian pyramid, coarse to fine (Bouguet's pyramidal LK). See lucas_kanade.cpp for the steps.
// Cost: O(levels · iterations · W·H·σ) (five separable Gaussian window sums per iteration).

#include <stop_token>
#include <vector>

#include <opencv2/core.hpp>

#include "core/common/task.hpp"
#include "core/flow/flow_common.hpp"

namespace vl::flow {

struct LucasKanadeParams {
    double sigma = 2.5;           // Gaussian window σ (px); the window is about 6σ wide
    int levels = 4;               // pyramid levels (1 = single scale)
    int iterations = 3;           // warping iterations per level
    double presmooth = 0.5;       // Gaussian blur of the frames before derivatives (px), 0 = none
    double minEigen = 0.0;        // pixels whose smallest eigenvalue is below this get NaN (0 = keep all)
    double tikhonov = 0.1;        // ε added to the diagonal of J (grey²/px²), keeps the solve defined
};

struct LucasKanadeResult {
    cv::Mat flow;         // CV_32FC2, NaN where rejected by minEigen
    cv::Mat minEigen;     // CV_32F, λ2 (smallest eigenvalue of J) at the finest level, grey²/px²
    cv::Mat maxEigen;     // CV_32F, λ1
    cv::Mat tensor;       // CV_32FC3, (G*Ix², G*IxIy, G*Iy²) at the finest level
    std::vector<double> meanUpdate;  // mean |Δw| per iteration (px, in full-resolution units), all levels in order
    int levelsUsed = 0;
};

// frame0, frame1: 8-bit grey or BGR images of the same size. onStep (optional) receives the
// full-resolution flow after every iteration.
LucasKanadeResult lucasKanade(const cv::Mat& frame0, const cv::Mat& frame1, const LucasKanadeParams& p,
                              std::stop_token stop = {}, const ProgressFn& progress = {},
                              const FlowCallback& onStep = {});

}  // namespace vl::flow
