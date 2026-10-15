#pragma once
// Building blocks shared by every optical-flow method. Course: "3D Vision and Motion", part 11.
//
// Notation. I0 and I1 are the two frames, w = (u, v) the flow, in pixels, frame 0 coordinates.
//
//   brightness constancy         I1(x + u, y + v) = I0(x, y)
//   linearised around a guess w0 (first-order Taylor expansion of I1 at x + w0):
//                                I1(x + w0) + Ix·(u − u0) + Iy·(v − v0) − I0(x) ≈ 0
//   the optical-flow constraint  Ix·du + Iy·dv + It = 0,   It = I1(x + w0) − I0(x)
//
// One equation, two unknowns per pixel: every method adds an assumption (a window of constant
// flow for Lucas–Kanade, a smooth field for Horn–Schunck). The Taylor expansion only holds for a
// motion of about a pixel, hence the two tools below that every method uses:
//   - warping: I1 resampled at x + w0 (warpImage), so that only the small remainder du is linearised;
//   - coarse to fine: the same problem on halved images (buildPyramid), where motions are halved too,
//     then the answer brought up a level (upsampleFlow) as the next level's guess w0.
//
// Flows are CV_32FC2 images (u, v). NaN marks "no estimate".

#include <functional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace vl::flow {

// Called by the iterative methods with the current estimate, already brought to full resolution,
// and a short description of the stage ("level 2 / 4, iteration 3"). Used to show convergence.
using FlowCallback = std::function<void(const cv::Mat& flow, const std::string& stage)>;

struct Derivatives {
    cv::Mat Ix, Iy, It;  // CV_32F
};

// Spatial derivatives with the 5-tap central difference
//     ∂I/∂x ≈ (I(x−2) − 8·I(x−1) + 8·I(x+1) − I(x+2)) / 12
// (exact on polynomials up to degree 4, much less biased than (I(x+1) − I(x−1)) / 2 on fine texture).
// When averageFrames is set, Ix and Iy are the mean of those of I0 and I1, as Horn and Schunck do:
// the constraint is then evaluated half-way between the frames, which makes it symmetric and
// noticeably more accurate. It = I1 − I0. Inputs: CV_32F, same size.
Derivatives derivatives(const cv::Mat& I0, const cv::Mat& I1, bool averageFrames = true);

// The constraint linearised around the current flow w: warp I1 by w, then Ix, Iy (averaged over
// I0 and the warped I1) and It = I1(x + w) − I0(x). A pixel whose warped position left the frame
// says nothing about the motion, so its three derivatives are set to zero.
Derivatives linearise(const cv::Mat& I0, const cv::Mat& I1, const cv::Mat& w);

// Gaussian pyramid (vl::pyramidDown at each step): level 0 is the image, level l is 2^l times
// smaller. Stops early when a side would fall under minSide pixels.
std::vector<cv::Mat> buildPyramid(const cv::Mat& img, int levels, int minSide = 16);

// Backward warp: out(x) = img(x + w(x)), bilinear (vl::bilinear). A NaN vector counts as zero
// motion. Pixels whose source falls outside the image get the nearest border value and are
// flagged in *outside (CV_8U, 255) when it is given. O(W·H).
cv::Mat warpImage(const cv::Mat& img, const cv::Mat& flow, cv::Mat* outside = nullptr);

// A flow of a coarser level brought to `size`, where one coarse pixel spans `factor` fine pixels
// (2 between neighbouring pyramid levels): the fine pixel x reads the coarse field at x / factor,
// bilinearly, and its vector is `factor` times longer.
cv::Mat upsampleFlow(const cv::Mat& flow, cv::Size size, float factor);

// Grey float frame in [0, 255], lightly blurred (sigma in px, 0 = none) so that derivatives are
// taken on a band-limited signal.
cv::Mat prepareFrame(const cv::Mat& img, double sigma);

// Mean |w| over the finite vectors, in pixels.
double meanMagnitude(const cv::Mat& flow);

}  // namespace vl::flow
