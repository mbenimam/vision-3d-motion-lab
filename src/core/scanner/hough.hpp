#pragma once
// The Hough transform for straight lines: every edge pixel votes for the lines that pass through it,
// and the lines with many votes are the long straight edges of the picture (the page's sides).
//
// A line is written in normal form  ρ = x cos θ + y sin θ,  θ ∈ [0°, 180°), ρ ∈ [−D, D] (D = the
// image diagonal): θ is the direction of its normal, ρ its signed distance to the origin. An edge
// pixel (x, y) lies on one line per θ, so it votes along a sinusoid of the (θ, ρ) accumulator; the
// sinusoids of collinear pixels cross at that line's (θ, ρ). Steps (hough.cpp):
//   1. accumulateVotes   each edge pixel votes for the θ near its gradient direction (± a window)
//   2. findPeaks         local maxima of the accumulator, then greedy non-maximum suppression
// Cost O(E · w) for E edge pixels and a window of w degrees, instead of O(E · 180) for blind voting.

#include <vector>

#include <opencv2/core.hpp>

namespace vl::scanner {

struct HoughLine {
    double theta = 0;  // radians, normal direction, in [0, π)
    double rho = 0;    // pixels
    int votes = 0;
};

struct HoughParams {
    int thetaWindowDeg = 12;  // an edge pixel votes for θ within this many degrees of its gradient direction
    int maxLines = 20;        // strongest peaks kept
    int suppressDeg = 6;      // two peaks closer than this in θ ...
    double suppressRho = 12;  // ... and this in ρ (pixels) are the same line
    int minVotes = 30;        // a line must be supported by at least this many edge pixels
};

struct HoughResult {
    cv::Mat votes;                 // CV_32S, rows = ρ bins (ρ + D, one per pixel), cols = 180 θ bins (one per degree)
    int rhoOffset = 0;             // D: row r is ρ = r − D
    std::vector<HoughLine> lines;  // strongest first
};

// edges: CV_8U (255 on edges); dx, dy: the gradient that produced them (gives each pixel's θ).
HoughResult houghLines(const cv::Mat& edges, const cv::Mat& dx, const cv::Mat& dy, const HoughParams& p);

// The point where two lines meet: solve [cos θ1 sin θ1; cos θ2 sin θ2] (x, y)ᵀ = (ρ1, ρ2)ᵀ.
// Returns false when they are (nearly) parallel.
bool intersect(const HoughLine& a, const HoughLine& b, cv::Point2d& out);

}  // namespace vl::scanner
