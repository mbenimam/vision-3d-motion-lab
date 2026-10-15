#pragma once
// Panorama by homographies. Course "3D Vision and Motion", part 2.
//
// Every image is mapped into the frame of a reference image (the middle one, so the distortion is
// shared on both sides) and blended on a common canvas. Steps (stitch.cpp):
//   1. chainToReference  with H_i : image i -> image i+1 between neighbours, the map of image i to the
//                        reference r is the product of the links in between (inverted right of r)
//   2. warpedBounds      the canvas: bounding box of every image's warped outline
//   3. warpInto          inverse mapping: for each canvas pixel p, x = (T H_i)⁻¹ p says where to read
//                        image i (bilinear), so the canvas has no holes
//   4. featherWeight     each image weighs its pixels by their distance to its own border, and
//   5. blend             the canvas is the weighted mean: seams between images of different exposure
//                        fade out over the overlap
//
// Cost O(canvas pixels · images).

#include <stop_token>
#include <vector>

#include <opencv2/core.hpp>

namespace vl::geometry {

// H_i maps image i -> image i+1 (n-1 links for n images). Returns the n maps image i -> reference.
std::vector<cv::Matx33d> chainToReference(const std::vector<cv::Matx33d>& links, int reference);

// Bounding box of the images' warped corners in the reference frame.
cv::Rect2d warpedBounds(const std::vector<cv::Size>& sizes, const std::vector<cv::Matx33d>& toReference);

struct StitchParams {
    bool feather = true;           // weighted mean; off: each image is pasted over the previous ones
    double maxCanvasPixels = 12e6; // a wild homography is shrunk to this budget instead of eating memory
};

struct Panorama {
    cv::Mat image;                             // 8-bit BGR canvas, black outside every image
    std::vector<cv::Matx33d> toCanvas;         // image i -> canvas pixel (translation/scale ∘ toReference)
    std::vector<std::vector<cv::Point2d>> outlines;  // warped border of each image on the canvas
    double scale = 1.0;                        // < 1 when the canvas had to be shrunk
};

// images: 8-bit grey or BGR. toReference: one map per image.
Panorama stitch(const std::vector<cv::Mat>& images, const std::vector<cv::Matx33d>& toReference,
                const StitchParams& p = {}, std::stop_token stop = {});

// The outline of a w x h image under H, sampled along its edges (straight under a homography, but
// sampling keeps it right when the image wraps behind the camera).
std::vector<cv::Point2d> warpedOutline(cv::Size size, const cv::Matx33d& H, int perEdge = 16);

}  // namespace vl::geometry
