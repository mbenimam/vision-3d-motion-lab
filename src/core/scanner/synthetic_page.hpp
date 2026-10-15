#pragma once
// Photos of a document with exact ground truth, for the scanner topic and its tests.
//
// A page (title, paragraphs, a figure, a table) is rendered flat, then placed on a desk with a
// known homography, lit unevenly, shadowed, blurred a little and given sensor noise. The truth is the
// page's four corners in the photo and the flat page itself. OpenCV draws and warps here: this file
// prepares inputs, it is not part of the method.

#include <cstdint>

#include <opencv2/core.hpp>

#include "core/scanner/quad.hpp"

namespace vl::scanner {

enum class DeskScene {
    Easy,               // mild perspective, wooden desk, soft light
    StrongPerspective,  // the phone held low: the far edge much shorter than the near one
    Shadowed,           // a hand's shadow across the page and a strong light falloff
    LowContrast,        // white paper on a pale desk
    Cluttered,          // a mug over one corner, a pen and a sticky note on the desk
};

struct DocumentPhoto {
    cv::Mat photo;  // 8-bit BGR, 960 × 1280
    cv::Mat page;   // 8-bit BGR, the flat page (850 × 1202, A4 proportions)
    Quad corners;   // the page's corners in the photo: the outer edges of its first/last pixels
};

// The flat page alone.
cv::Mat renderPage(cv::Size size = {850, 1202});

DocumentPhoto documentPhoto(DeskScene scene, std::uint32_t seed = 7);

}  // namespace vl::scanner
