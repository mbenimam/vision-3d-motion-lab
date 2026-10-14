#pragma once
// Cleaning a per-pixel foreground mask with its neighbours: moving objects are compact blobs, sensor
// noise is isolated pixels. Course: "3D Vision and Motion", part 9 (space-time regularisation).
//
// Morphology (alternatingSequentialFilter): opening then closing with discs of growing radius
// 1, 2, ..., order. An opening (erode, then dilate) removes specks smaller than the disc; a closing
// (dilate, then erode) fills holes smaller than it. Built on vl::open / vl::close: O(W·H·r²).
//
// Markov random field (icmPotts): the label image x (0 background, 1 foreground) is the minimum of
//
//     U(x) = ½ Σ_s (r_s − 2 x_s)²  +  β Σ_{s~t} [x_s ≠ x_t]
//            ─── data term ──────     ─── Potts prior, 8 neighbours, each pair once ───
//
// where r_s = |I − B| / T is the detection margin (1 on the threshold, clipped to [0, rMax]). With
// β = 0 the minimum is exactly the per-pixel test r > 1; β > 0 makes a label pay for each neighbour
// that disagrees. It is the course's Gibbs energy (Potts ±β, Gaussian data term with α = 2 on the
// normalised difference) up to constants. Minimised by ICM (iterated conditional modes):
//   1. localEnergies   for one pixel, U restricted to the terms that involve it, for x_s = 0 and 1
//   2. icmSweep        visit every pixel in raster order and give it the label of lower local energy
//   3. icmPotts        sweep until a sweep changes nothing (or maxSweeps)
// Each move can only lower U, so U never increases and ICM stops at a local minimum. One sweep costs
// O(W·H·8); a few sweeps are usually enough.

#include <stop_token>
#include <vector>

#include <opencv2/core.hpp>

namespace vl::motion {

// Opening then closing with discs of radius 1, 2, ..., order. mask: CV_8U 0/255.
cv::Mat alternatingSequentialFilter(const cv::Mat& mask, int order = 2);

struct MrfParams {
    double beta = 1.0;   // β: price of one disagreeing pair of neighbours
    double rMax = 4.0;   // clip of the margin, so one very strong pixel cannot dominate its region
    int maxSweeps = 10;
};

struct MrfResult {
    cv::Mat mask;                // CV_8U 0/255
    long changes = 0;            // pixels whose label differs from the initial mask
    int sweeps = 0;
    std::vector<double> energy;  // U before the first sweep, then after each sweep
};

// margin: CV_32F detection margin r = |I − B| / T; init: CV_8U 0/255 starting labels (usually r > 1).
MrfResult icmPotts(const cv::Mat& margin, const cv::Mat& init, const MrfParams& p, std::stop_token stop = {});

// U(x) as defined above, for checking and plotting.
double mrfEnergy(const cv::Mat& margin, const cv::Mat& labels, const MrfParams& p);

// r = difference / max(threshold, 1): the margin the MRF data term reads.
cv::Mat detectionMargin(const cv::Mat& difference, const cv::Mat& threshold);

}  // namespace vl::motion
