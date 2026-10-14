#pragma once
// Horn–Schunck (1981), a global variational optical flow. Course: "3D Vision and Motion", part 11.
//
// Instead of trusting a window, look for the field w = (u, v) that minimises one energy over the
// whole image, data term (the linearised brightness constancy) plus smoothness weighted by α²:
//
//     E(u, v) = Σ (Ix·u + Iy·v + It)²  +  α² Σ ( |∇u|² + |∇v|² )
//
// Where the image is flat, or an edge only gives the normal component, the smoothness term fills
// the flow in from the neighbours: no aperture hole, at the price of blurring motion boundaries.
// The minimiser solves the Euler–Lagrange equations, a sparse linear system iterated by Jacobi or
// Gauss–Seidel sweeps (equations in horn_schunck.cpp). Like Lucas–Kanade, it runs coarse to fine
// with re-warping. Cost: O(levels · iterations · W·H).

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
    int warps = 2;                // re-linearisations per level (the sweeps are split between them)
    double presmooth = 1.0;       // Gaussian blur of the frames before derivatives (px)
    HornSchunckSolver solver = HornSchunckSolver::GaussSeidel;
    int reportEvery = 10;         // call onStep every this many sweeps
};

struct HornSchunckResult {
    cv::Mat flow;                      // CV_32FC2
    cv::Mat dataWeight;                // CV_32F in [0, 1]: |∇I|² / (|∇I|² + α²) at the finest level
    std::vector<double> energy;        // E per pixel after each sweep at the finest level
    std::vector<double> meanUpdate;    // mean |Δw| per sweep (full-resolution px), all levels
    int sweeps = 0;
    int levelsUsed = 0;
};

HornSchunckResult hornSchunck(const cv::Mat& frame0, const cv::Mat& frame1, const HornSchunckParams& p,
                              std::stop_token stop = {}, const ProgressFn& progress = {},
                              const FlowCallback& onStep = {});

// Lower-level pieces, exposed for the tests. It0 is the data offset of the linearisation, so that
// the residual of a pixel is Ix·u + Iy·v + It0 (It0 = It − Ix·u0 − Iy·v0 after warping by w0).
// E above, per pixel.
double hornSchunckEnergy(const cv::Mat& u, const cv::Mat& v, const cv::Mat& Ix, const cv::Mat& Iy,
                         const cv::Mat& It0, double alpha);
// One sweep over all pixels, in place (Gauss–Seidel) or from a copy (Jacobi). Returns the mean |Δw|.
double hornSchunckSweep(cv::Mat& u, cv::Mat& v, const cv::Mat& Ix, const cv::Mat& Iy, const cv::Mat& It0,
                        double alpha, HornSchunckSolver solver);

}  // namespace vl::flow
