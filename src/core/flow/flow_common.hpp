#pragma once
// Building blocks shared by every optical-flow method: image derivatives, Gaussian pyramids and
// warping. Notation: I0 and I1 are the two frames, w = (u, v) the flow, so that the brightness
// constancy assumption reads I1(x + w(x)) = I0(x). Linearised around a current guess w0 it gives
// the optical-flow constraint  Ix·du + Iy·dv + It = 0  with It = I1(x + w0) - I0(x).
//
// Flows are CV_32FC2 images (u, v) in pixels, frame 0 coordinates. NaN marks "no estimate".

#include <functional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace vl::flow {

// Called by the iterative methods with the current estimate, already brought to full resolution,
// and a short description of the stage ("level 2 / 4, iteration 30"). Used to show convergence.
using FlowCallback = std::function<void(const cv::Mat& flow, const std::string& stage)>;

struct Derivatives {
    cv::Mat Ix, Iy, It;  // CV_32F
};

// Spatial derivatives with the 5-tap central difference [1 -8 0 8 -1] / 12 (exact for cubics, much
// less biased than [-1 0 1] / 2 on fine texture). When averageFrames is set the spatial derivatives
// are the mean of those of I0 and I1 (as Horn and Schunck do): the constraint is then evaluated
// half-way between the frames, which makes it symmetric and noticeably more accurate.
// It = I1 - I0. Inputs are CV_32F of the same size.
Derivatives derivatives(const cv::Mat& I0, const cv::Mat& I1, bool averageFrames = true);

// Gaussian pyramid: level 0 is the image itself, each next level is blurred and halved (cv::pyrDown).
// Stops early when a side would fall under minSide pixels.
std::vector<cv::Mat> gaussianPyramid(const cv::Mat& img, int levels, int minSide = 16);

// Backward warp of a float image: out(x) = img(x + flow(x)), bilinear. Pixels whose source falls
// outside the image are flagged in *outside (CV_8U, 255) when it is given; their values are clamped.
cv::Mat warpImage(const cv::Mat& img, const cv::Mat& flow, cv::Mat* outside = nullptr);

// Resize a flow field to `size`, scaling the vectors by the same factors (coarse -> fine level).
cv::Mat resizeFlow(const cv::Mat& flow, cv::Size size);

// Grey float frames prepared for the estimators: converted to [0, 255] floats and lightly blurred
// (sigma in pixels, 0 = none) so that derivatives are taken on a band-limited signal.
cv::Mat prepareFrame(const cv::Mat& img, double sigma);

// Mean magnitude of a flow field (skipping NaN), in pixels.
double meanMagnitude(const cv::Mat& flow);

}  // namespace vl::flow
