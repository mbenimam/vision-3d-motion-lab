#pragma once
// From candidate lines to the page's four corners.
//
// The Hough peaks (hough.hpp) are the long straight edges of the photo: the page's sides, but also
// lines of text, the desk's edge, a pen. A page is the quadrilateral formed by four of them that is
// convex, has sensible angles, is large, and whose four sides really are edges along their whole
// length. Steps (quad.cpp):
//   1. quadFromLines       four lines and a choice of which ones are opposite → four corners
//   2. isPlausible         convex, every angle in [35°, 145°], area ≥ a share of the photo
//   3. sideSupport         share of each side's length that lies on an edge of the right orientation
//   4. bestQuadrilateral   maximise area × Π support over every 4-line combination
//   5. refineSide          fit each chosen side again at full resolution, to a fraction of a pixel
//   6. orderCorners        top-left, top-right, bottom-right, bottom-left
// Cost of step 4: C(n, 4) · 3 candidates (n ≤ 20 lines: ≤ 14 535), each checked in O(perimeter).

#include <array>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "core/scanner/edges.hpp"
#include "core/scanner/hough.hpp"

namespace vl::scanner {

// Corners in the order top-left, top-right, bottom-right, bottom-left (clockwise on screen).
using Quad = std::array<cv::Point2d, 4>;

struct QuadParams {
    double minAreaShare = 0.08;  // the page covers at least this share of the photo
    double minSupport = 0.45;    // each side must lie on edges for at least this share of its length
    double minAngleDeg = 35;     // interior angles of the page seen in perspective
    double maxAngleDeg = 145;
    double margin = 0.08;  // corners may fall this share of the size outside the photo
};

struct QuadCandidate {
    Quad corners;                     // ordered
    std::array<double, 4> support{};  // step 3, side i runs from corner i to corner i + 1, in [0, 1]
    std::array<size_t, 4> lines{};    // indices of the four Hough lines used (in cycle order)
    double area = 0, score = 0;
};

// Step 4 (uses 1-3). Lines and edges at the detection scale. nullopt when no plausible page exists.
std::optional<QuadCandidate> bestQuadrilateral(const std::vector<HoughLine>& lines, const EdgeMap& edges,
                                               const QuadParams& p);

// Step 5: the line through the side a → b, re-fitted on the full-resolution grey image (CV_32F):
// the position of the strongest step across the side, found to sub-pixel accuracy at points along it,
// then a total least squares line through them with outliers (an object over the edge) removed.
HoughLine refineSide(const cv::Mat& grey, cv::Point2d a, cv::Point2d b);

// Step 6: sort by angle around the centroid (clockwise on screen, y pointing down), then start at
// the corner nearest the photo's top-left (smallest x + y).
Quad orderCorners(const std::vector<cv::Point2d>& points);

// Polygon area by the shoelace formula, |Σ (x_i y_{i+1} − x_{i+1} y_i)| / 2.
double quadArea(const Quad& q);

}  // namespace vl::scanner
