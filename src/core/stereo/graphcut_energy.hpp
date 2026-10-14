#pragma once
// The energy minimised by graph-cut stereo with occlusions (Kolmogorov and Zabih, "Computing visual
// correspondence with occlusions using graph cuts", ICCV 2001). Course part 8.
//
// The unknowns are *assignments* a = <p, p'>: a left pixel p = (x, y) and a right pixel p' = (x − d, y)
// on the same row, at disparity d(a) = d. A configuration f makes some of them active. It is *unique*
// when every pixel, left or right, is in at most one active assignment; a pixel in none is occluded.
// For unique configurations
//
//   E(f) = E_data(f) + E_occ(f) + E_smooth(f)
//   E_data   = Σ_{a active} D(a)                          D(a): how different p and p' look
//   E_occ    = Σ_{p in L or R, p occluded} K               K: the price of an unmatched pixel
//   E_smooth = Σ_{{a1,a2} ∈ N} V(a1,a2) · [exactly one of a1, a2 active]
//
// and E = +∞ for non-unique ones. N pairs assignments of the *same* disparity whose left pixels are
// 4-neighbours, so a depth edge (neighbours at different disparities) pays V on both sides.
// The course writes the occlusion cost C_p; here it is the same K for every pixel.
//
// With modelOcclusions = false the same file gives the classic Potts labelling energy (Boykov,
// Veksler, Zabih 2001): every left pixel has a disparity, E = Σ_p D(p, d_p) + Σ_{p~q} V·[d_p ≠ d_q].

#include <cstdint>

#include <opencv2/core.hpp>

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
    double edgeThreshold = 8;   // grey levels
    double occlusionCost = 15;  // K, per occluded pixel (left or right)
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

constexpr int kOccluded = -1;  // "no active assignment" in a Configuration

// A unique configuration, stored from both sides: the disparity of each left pixel's active
// assignment and of each right pixel's (CV_32S, kOccluded where none). In Potts mode `right` is
// empty and every left pixel has a disparity.
struct Configuration {
    cv::Mat left, right;
};

struct EnergyTerms {
    double data = 0, occlusion = 0, smoothness = 0;
    double total() const { return data + occlusion + smoothness; }
};

// The terms D, K, V for one stereo pair, and E(f) for a whole configuration.
class StereoEnergy {
public:
    StereoEnergy(const cv::Mat& left, const cv::Mat& right, const GraphCutParams& p);

    const GraphCutParams& params() const { return p_; }
    int width() const { return L_.cols; }
    int height() const { return L_.rows; }

    // Whether the assignment <(x, y), (x − d, y)> exists: its right pixel is inside the image.
    bool exists(int x, int d) const { return x - d >= 0 && x - d < width(); }

    double D(int x, int y, int d) const;  // data cost of <(x, y), (x − d, y)>
    double K() const { return p_.occlusionCost; }
    // V for the assignments of disparity d at the neighbouring left pixels (x0, y0) and (x1, y1).
    double V(int x0, int y0, int x1, int y1, int d) const;

    EnergyTerms evaluate(const Configuration& f) const;

private:
    double dissimilarity(int x, int y, int d) const;
    double dataTerm(const Configuration& f) const;
    double occlusionTerm(const Configuration& f) const;
    double smoothnessTerm(const Configuration& f) const;

    GraphCutParams p_;
    cv::Mat L_, R_;                  // CV_32FC3 colour values
    cv::Mat Lmin_, Lmax_, Rmin_, Rmax_;  // CV_32FC3 Birchfield–Tomasi intervals
    cv::Mat Lgrey_, Rgrey_;          // CV_32F intensities, for the adaptive λ
};

}  // namespace vl::stereo
