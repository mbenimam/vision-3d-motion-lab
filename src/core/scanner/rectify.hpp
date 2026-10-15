#pragma once
// Flattening the page: the homography from its four corners to an upright rectangle, and the image
// resampled through it. Course "3D Vision and Motion", part 2 (homographies and the DLT).
//
//   1. outputSize       W × H of the flat page: the longer of each pair of opposite sides, or A4 (1 : √2)
//   2. pageHomography   H with λ (u, v, 1)ᵀ = H (x, y, 1)ᵀ, taking corner i of the photo to corner i of
//                       the rectangle; four points give H exactly (vl::geometry::dltHomography)
//   3. warpInverse      for every output pixel (u, v): (x, y) = H⁻¹ (u, v), sample the photo there with
//                       bilinear interpolation. Cost O(W·H) whatever the photo size.
//
// Why inverse: pushing every photo pixel forward through H leaves holes where the page is stretched
// and collisions where it is squeezed; pulling every output pixel back gives each one exactly one value.

#include <opencv2/core.hpp>

#include "core/scanner/quad.hpp"

namespace vl::scanner {

enum class PageRatio { Auto, A4 };

// Step 1. The result is limited to maxSide pixels on its longer side.
cv::Size outputSize(const Quad& corners, PageRatio ratio, int maxSide = 1600);

// The rectangle's corners in pixel coordinates, in the order of Quad: the outer edges of the
// output pixels, (−½, −½) to (W − ½, H − ½), so that pixel centres sit at integers.
Quad rectangleCorners(cv::Size size);

// Step 2: photo → rectangle.
cv::Matx33d pageHomography(const Quad& corners, cv::Size size);

// Step 3. photo: 8-bit grey or BGR; H: photo → output. The result has the photo's type.
cv::Mat warpInverse(const cv::Mat& photo, const cv::Matx33d& H, cv::Size size);

// Steps 2 and 3 together.
cv::Mat rectify(const cv::Mat& photo, const Quad& corners, cv::Size size);

}  // namespace vl::scanner
