#pragma once
// Global stereo with occlusions by graph cuts (Kolmogorov and Zabih, ICCV 2001): minimise the energy
// of graphcut_energy.hpp by α-expansion moves, each solved exactly by one minimum cut (maxflow.hpp).
//
// An α-expansion from configuration f may switch off any active assignment and switch on any
// assignment of disparity α. Each move is a binary problem (keep or drop, on or off), its energy is
// regular, so a minimum cut gives the best move. Moves for every α are tried in turn until a whole
// pass lowers nothing. Cost of a move: a graph of at most 2·W·H nodes and 6·W·H edges, solved in close
// to linear time; a pass is one move per disparity.
//
// With modelOcclusions = false the same loop minimises the Potts labelling energy (Boykov, Veksler,
// Zabih 2001), which gives every left pixel a disparity: comparing the two shows what the
// assignment formulation buys at depth edges.

#include <functional>
#include <stop_token>
#include <vector>

#include <opencv2/core.hpp>

#include "core/common/task.hpp"
#include "core/stereo/graphcut_energy.hpp"
#include "core/stereo/maxflow.hpp"

namespace vl::stereo {

// State after each expansion move, for progress displays.
struct GraphCutStep {
    int iteration = 0;      // pass number, from 1
    int alpha = 0;          // the disparity just expanded
    int move = 0;           // number of moves tried so far, from 1
    bool improved = false;  // whether this move lowered the energy
    double energy = 0;
    const cv::Mat* disparity = nullptr;  // CV_32F, NaN where occluded
};
using GraphCutObserver = std::function<void(const GraphCutStep&)>;

struct GraphCutResult {
    cv::Mat disparity;           // CV_32F, left coordinates, NaN where occluded
    cv::Mat occluded;            // CV_8U, 255 where the left pixel has no match
    cv::Mat occludedRight;       // CV_8U, 255 where the right pixel has no match
    std::vector<double> energy;  // energy[0] = initial configuration, then one value per move
    EnergyTerms terms;           // final split of the energy
    int moves = 0;               // expansion moves tried
    int improvingMoves = 0;      // moves that lowered the energy
    int iterations = 0;          // passes over the disparities
    bool converged = false;      // a whole pass without improvement
};

// The solver, one move at a time. Starts, as in the paper, from the configuration with no active
// assignment (every pixel occluded); in Potts mode from disparity minDisparity everywhere.
class GraphCutStereo {
public:
    GraphCutStereo(const cv::Mat& left, const cv::Mat& right, const GraphCutParams& p);

    // The best configuration within one α-expansion of the current one (one minimum cut). Keeps it
    // and returns true when it lowers the energy; otherwise leaves the configuration unchanged.
    bool expand(int alpha);

    double energy() const { return energyTerms().total(); }
    EnergyTerms energyTerms() const { return E_.evaluate(f_); }
    cv::Mat disparity() const;      // CV_32F, NaN where occluded
    cv::Mat occludedRight() const;  // CV_8U, 255 where the right pixel has no match
    // Replace the configuration (CV_32F, NaN = occluded, rounded to integers). Returns false and
    // keeps the old one if the map is not unique or uses a disparity outside the range.
    bool setDisparity(const cv::Mat& disparity);

private:
    // The Kolmogorov–Zabih move, in the order it is built (graphcut_stereo.cpp).
    void addAssignmentNodes(int alpha);
    void addOcclusionTerms(int alpha);
    void addOcclusionTerm(int current, int candidate);
    void addSmoothnessTerms(int alpha);
    void addSmoothnessTerm(int x0, int y0, int x1, int y1, int alpha);
    void applyAssignmentMove(int alpha);
    // The Potts move.
    void buildPottsMove(int alpha);
    void applyPottsMove(int alpha);

    void rebuildRight();
    bool onSinkSide(int node) const { return graph_.side(node) == MaxFlow::Side::Sink; }

    StereoEnergy E_;
    Configuration f_;
    MaxFlow graph_;
    cv::Mat nodeCurrent_;  // CV_32S, per left pixel: node of its current assignment in the move, or -1
    cv::Mat nodeAlpha_;    // CV_32S, per left pixel: node of its α-assignment, or -1
};

// left, right: rectified 8-bit grey or BGR images of the same size; left (x, y) matches right (x − d, y).
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
