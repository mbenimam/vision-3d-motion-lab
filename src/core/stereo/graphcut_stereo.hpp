#pragma once
// Global stereo by graph cuts, after Kolmogorov and Zabih, "Computing visual correspondence with
// occlusions using graph cuts", ICCV 2001.
//
// The unknowns are not disparities of left pixels but *assignments*: pairs a = <p, q> of a left
// pixel p and a right pixel q on the same row, at disparity d(a) = px - qx. A configuration makes
// some assignments active; it is *unique* when every pixel, left or right, is in at most one active
// assignment, and a pixel in none is occluded. The energy of a unique configuration is
//
//   E = Σ_{a active} D(a)                       data: how different p and q look
//     + Σ_{p ∈ L ∪ R, p occluded} K              occlusion: each unmatched pixel costs K
//     + Σ_{{a1,a2} ∈ N} V(a1,a2)·[a1 active xor a2 active]   smoothness
//
// where N pairs assignments of the same disparity whose left pixels are 4-neighbours. Non-unique
// configurations have infinite energy, so uniqueness (and with it the occlusions) is enforced by
// construction, symmetrically in both images.
//
// Minimisation by α-expansion: for one disparity α at a time, find the best configuration that
// may switch off current assignments and switch on assignments of disparity α. Each such move is a
// binary problem: one variable per current assignment (keep / drop) and per α-assignment (on / off).
// All its terms are submodular, so one minimum cut (our Boykov–Kolmogorov max-flow) gives the
// optimal move exactly. Passes over all disparities repeat until none lowers the energy.
//
// Cost of one move: a graph of at most 2·W·H nodes and about 6·W·H edges; the max-flow is close to
// linear on such grids. A pass is one move per disparity.
//
// With modelOcclusions = false the same machinery minimises the classic Potts labelling energy of
// Boykov, Veksler and Zabih (PAMI 2001): one disparity for every left pixel, no occlusion term, no
// uniqueness. Comparing the two shows what the assignment formulation buys at depth edges.

#include <cstdint>
#include <functional>
#include <memory>
#include <stop_token>
#include <vector>

#include <opencv2/core.hpp>

#include "core/common/task.hpp"

namespace vl::stereo {

enum class DataCost { Absolute, Squared };

struct GraphCutParams {
    int minDisparity = 0;
    int maxDisparity = 16;  // inclusive
    // Smoothness: V = λ between neighbours of similar intensity; with adaptiveLambda, V = λ/3 across an
    // intensity edge (the two left pixels or the two right pixels differ by >= edgeThreshold), so depth
    // discontinuities are cheaper where the image has an edge. (The paper prints the two cases of its
    // eq. 13 the other way round, which contradicts its own stated intent; we follow the intent.)
    double lambda = 6;
    bool adaptiveLambda = true;
    double edgeThreshold = 8;  // grey levels
    double occlusionCost = 15; // K, per occluded pixel (left or right)
    // Data term: per-channel dissimilarity δ (Birchfield–Tomasi or plain |ΔI|), averaged over channels,
    // truncated at `truncation`; Absolute: D = min(δ, T); Squared: D = min(δ, T)² / T. Both span [0, T],
    // so λ and K keep the same meaning when switching.
    DataCost cost = DataCost::Squared;
    double truncation = 30;
    bool birchfieldTomasi = true;
    bool modelOcclusions = true;  // false: Potts labelling of the left image (no occlusions)
    int maxIterations = 4;        // passes over all disparities
    bool randomOrder = true;
    std::uint32_t seed = 1;
};

// State after each expansion move, for progress displays.
struct GraphCutStep {
    int iteration = 0;          // pass number, from 1
    int alpha = 0;              // the disparity just expanded
    int move = 0;               // number of moves tried so far, from 1
    bool improved = false;      // whether this move lowered the energy
    double energy = 0;
    const cv::Mat* disparity = nullptr;  // CV_32F, NaN where occluded
};
using GraphCutObserver = std::function<void(const GraphCutStep&)>;

struct GraphCutResult {
    cv::Mat disparity;            // CV_32F, left coordinates, NaN where occluded
    cv::Mat occluded;             // CV_8U, 255 where the left pixel has no match
    cv::Mat occludedRight;        // CV_8U, 255 where the right pixel has no match
    std::vector<double> energy;   // energy[0] = initial configuration, then one value per move
    double dataEnergy = 0, occlusionEnergy = 0, smoothEnergy = 0;  // final split of the energy
    int moves = 0;                // expansion moves tried
    int improvingMoves = 0;       // moves that lowered the energy
    int iterations = 0;           // passes over the disparities
    bool converged = false;       // a whole pass without improvement
};

// The solver, one move at a time. Starts from the configuration with no active assignment (every
// pixel occluded), as in the paper; in Potts mode from disparity minDisparity everywhere.
class GraphCutStereo {
public:
    GraphCutStereo(const cv::Mat& left, const cv::Mat& right, const GraphCutParams& p);
    ~GraphCutStereo();
    GraphCutStereo(const GraphCutStereo&) = delete;
    GraphCutStereo& operator=(const GraphCutStereo&) = delete;

    // The optimal configuration within one α-expansion of the current one (one min cut). Keeps it and
    // returns true when it has a lower energy, otherwise leaves the configuration unchanged.
    bool expand(int alpha);

    double energy() const;  // of the current configuration
    void energyTerms(double& data, double& occlusion, double& smooth) const;
    cv::Mat disparity() const;      // CV_32F, NaN where occluded
    cv::Mat occludedRight() const;  // CV_8U
    // Replace the configuration (CV_32F, NaN = occluded, rounded to integers). Returns false and
    // keeps the old one if the map is not unique or uses a disparity outside the range.
    bool setDisparity(const cv::Mat& disparity);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// left, right: rectified 8-bit grey or BGR images of the same size; left (x, y) matches right (x - d, y).
GraphCutResult graphCutStereo(const cv::Mat& left, const cv::Mat& right, const GraphCutParams& p,
                              std::stop_token stop = {}, const ProgressFn& progress = {},
                              const GraphCutObserver& observer = {});

// Energy of a given left disparity map (CV_32F, NaN = occluded; values rounded to integers) under
// the model of p. With occlusions, the map must be unique (no two left pixels on one right pixel);
// the right image's occlusions follow from it. Returns +inf for a non-unique map.
double graphCutEnergy(const cv::Mat& left, const cv::Mat& right, const cv::Mat& disparity, const GraphCutParams& p);

// Occlusion mask implied by a ground-truth disparity map (CV_32F, 0 = unknown): a left pixel is
// hidden in the right view when a pixel further right on the row, hence in front, lands to the left
// of its match. CV_8U, 255 = occluded; unknown pixels are not marked.
cv::Mat occlusionsFromDisparity(const cv::Mat& truth);

}  // namespace vl::stereo
