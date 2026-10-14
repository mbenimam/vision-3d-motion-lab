#pragma once
// Dense Lucas–Kanade (1981): a local method. One constraint Ix·u + Iy·v + It = 0 per pixel cannot
// fix two unknowns, so assume the flow is constant over a Gaussian window G_σ and solve, at every
// pixel, the 2x2 least-squares system
//
//     [ G*Ix²   G*IxIy ] [du]     [ G*IxIt ]
//     [ G*IxIy  G*Iy²  ] [dv] = - [ G*IyIt ]         (J · dw = -b, J the structure tensor)
//
// in closed form. The eigenvalues of J say how well the window constrains the motion: both large on
// a corner or texture, one ~0 along a straight edge (only the normal component is measurable: the
// aperture problem), both ~0 on a flat area. The smallest eigenvalue is returned as a confidence.
// A tiny Tikhonov term keeps the solve defined; on an edge it then returns the normal flow.
//
// Linearisation only holds for motions of about a pixel, hence:
//  - iterations with warping (Gauss–Newton): re-warp I1 by the current flow, solve for the increment;
//  - coarse to fine (Bouguet's pyramidal LK): estimate on a halved, blurred copy where the motion is
//    halved too, upsample, refine on the next level.
// Cost: O(levels · iterations · W·H·σ) (separable Gaussian sums of five products per iteration).

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
    double tikhonov = 0.1;        // ε added to the diagonal of J (grey²/px²)
};

struct LucasKanadeResult {
    cv::Mat flow;         // CV_32FC2, NaN where rejected by minEigen
    cv::Mat minEigen;     // CV_32F, λ2 (smallest eigenvalue of J) at the finest level, grey²/px²
    cv::Mat maxEigen;     // CV_32F, λ1
    cv::Mat tensor;       // CV_32FC3, (Ix², IxIy, Iy²) summed over the window at the finest level
    std::vector<double> meanUpdate;  // mean |dw| per iteration (px, in full-resolution units), all levels in order
    int levelsUsed = 0;
};

// frame0, frame1: 8-bit grey or BGR images of the same size. onStep (optional) receives the
// full-resolution flow after every iteration.
LucasKanadeResult lucasKanade(const cv::Mat& frame0, const cv::Mat& frame1, const LucasKanadeParams& p,
                              std::stop_token stop = {}, const ProgressFn& progress = {},
                              const FlowCallback& onStep = {});

}  // namespace vl::flow
