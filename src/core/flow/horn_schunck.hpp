#pragma once
// Horn–Schunck (1981): a global method. Instead of trusting a window, look for the field w = (u, v)
// that minimises one energy over the whole image,
//
//     E(w) = Σ (Ix·u + Iy·v + It)²  +  α² Σ ( |∇u|² + |∇v|² ),
//
// data term (brightness constancy, linearised) plus smoothness weighted by α². Where the image is
// flat or an edge only gives the normal component, the smoothness term fills the flow in from the
// neighbours: no aperture hole, at the price of blurring motion boundaries.
//
// Discretisation: ∇ by finite differences with Horn and Schunck's weights (1/6 for the 4 direct
// neighbours, 1/12 for the diagonals). Setting ∂E/∂(u, v) = 0 at one pixel with its neighbours
// fixed gives the classic update, solved jointly for (u, v):
//
//     t = (Ix·ū + Iy·v̄ + It) / (α²·s + Ix² + Iy²),   u = ū - Ix·t,   v = v̄ - Iy·t
//
// with ū, v̄ the weighted neighbour means and s the sum of the neighbour weights (1 inside the
// image). Jacobi applies it to all pixels from the previous field; Gauss–Seidel uses the new values
// as soon as they exist, which is exact block coordinate descent: E never increases, and it
// converges about twice as fast. Like Lucas–Kanade, it is run coarse to fine with re-warping.
// Cost: O(levels · iterations · W·H).

#include <stop_token>
#include <vector>

#include <opencv2/core.hpp>

#include "core/common/task.hpp"
#include "core/flow/flow_common.hpp"

namespace vl::flow {

enum class HornSchunckSolver { Jacobi, GaussSeidel };

struct HornSchunckParams {
    double alpha = 12.0;          // smoothness weight α (grey levels per px of flow gradient)
    int iterations = 150;         // sweeps per pyramid level
    int levels = 4;               // pyramid levels (1 = single scale)
    int warps = 3;                // re-linearisations per level (the sweeps are split between them)
    double presmooth = 1.0;       // Gaussian blur of the frames before derivatives (px)
    HornSchunckSolver solver = HornSchunckSolver::GaussSeidel;
    int reportEvery = 10;         // call onStep every this many sweeps
};

struct HornSchunckResult {
    cv::Mat flow;                      // CV_32FC2
    cv::Mat dataWeight;                // CV_32F in [0, 1]: |∇I|² / (|∇I|² + α²·s) at the finest level
    std::vector<double> energy;        // E per pixel after each sweep at the finest level
    std::vector<double> meanUpdate;    // mean |Δw| per sweep (full-resolution px), all levels
    int sweeps = 0;
    int levelsUsed = 0;
};

HornSchunckResult hornSchunck(const cv::Mat& frame0, const cv::Mat& frame1, const HornSchunckParams& p,
                              std::stop_token stop = {}, const ProgressFn& progress = {},
                              const FlowCallback& onStep = {});

// Lower-level pieces, exposed for the tests.
// The energy above per pixel, for the linearisation data r = Ix·u + Iy·v + It0 (It0 already
// accounts for the flow the frames were warped by).
double hornSchunckEnergy(const cv::Mat& u, const cv::Mat& v, const cv::Mat& Ix, const cv::Mat& Iy,
                         const cv::Mat& It0, double alpha);
// One sweep in place (Gauss–Seidel) or from a copy (Jacobi). Returns the mean |Δw|.
double hornSchunckSweep(cv::Mat& u, cv::Mat& v, const cv::Mat& Ix, const cv::Mat& Iy, const cv::Mat& It0,
                        double alpha, HornSchunckSolver solver);

}  // namespace vl::flow
