#pragma once
// The image operations every method in this library is built from, written out so that nothing
// between the pixels and the algorithm is hidden. All work on single-channel CV_32F images
// (use toGrayF) and treat the image border by replicating the edge pixels.
//
//   convolve      g(x, y) = Σ_{i,j} k(i, j) · f(x + i, y + j)              (correlation form, see below)
//   separable     a 2D kernel k(i, j) = a(j) · b(i) costs 2(2r+1) per pixel instead of (2r+1)²
//   gaussian      G_σ(t) = exp(−t² / 2σ²) / Σ, truncated at 3σ
//   sobel         ∂f/∂x ≈ [1 2 1]ᵀ ⊗ [−1 0 1] / 8, ∂f/∂y likewise (the 1/8 makes a unit ramp give 1)
//   erode/dilate  minimum / maximum over a disc of radius r
//   pyramid       blur with σ = 1, keep every other row and column
//
// Note on "convolution": image processing usually applies the kernel without flipping it
// (correlation). All kernels here are symmetric or written for that form, so the difference never
// matters; we say "convolve" as the course does.

#include <vector>

#include <opencv2/core.hpp>

namespace vl {

// Sample f at (x, y) with edge replication: the image is extended by repeating its border.
inline float at(const cv::Mat& f, int y, int x) {
    y = y < 0 ? 0 : (y >= f.rows ? f.rows - 1 : y);
    x = x < 0 ? 0 : (x >= f.cols ? f.cols - 1 : x);
    return f.at<float>(y, x);
}

// Bilinear interpolation at a real position (x, y), edge replicated.
float bilinear(const cv::Mat& f, float x, float y);

// 2D correlation with any (2r+1)×(2r+1) CV_32F kernel. O(W·H·(2r+1)²).
cv::Mat convolve(const cv::Mat& f, const cv::Mat& kernel);

// Rows filtered with `horizontal`, then columns with `vertical` (both odd-length, centred). O(W·H·(2r+1)).
cv::Mat convolveSeparable(const cv::Mat& f, const std::vector<float>& horizontal, const std::vector<float>& vertical);

// Normalised 1D Gaussian of standard deviation sigma, radius ceil(3σ).
std::vector<float> gaussianKernel(double sigma);

// Gaussian blur: the separable product of two 1D Gaussians.
cv::Mat gaussianBlur(const cv::Mat& f, double sigma);

// Image derivatives with the Sobel operator, scaled so that f(x, y) = x gives dx = 1 everywhere.
struct Gradient {
    cv::Mat dx, dy;  // CV_32F
};
Gradient sobel(const cv::Mat& f);

// Morphology on a binary CV_8U mask (0 or 255) with a disc of radius r.
cv::Mat erode(const cv::Mat& mask, int r);
cv::Mat dilate(const cv::Mat& mask, int r);
cv::Mat open(const cv::Mat& mask, int r);   // erode then dilate: removes specks smaller than the disc
cv::Mat close(const cv::Mat& mask, int r);  // dilate then erode: fills holes smaller than the disc

// One pyramid step: blur (σ = 1) then keep every other pixel. Size becomes ceil(W/2) × ceil(H/2).
cv::Mat pyramidDown(const cv::Mat& f);
// Levels[0] is the image itself; each next level is pyramidDown of the previous one.
std::vector<cv::Mat> gaussianPyramid(const cv::Mat& f, int levels);

}  // namespace vl
