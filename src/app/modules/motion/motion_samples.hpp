#pragma once
// Inputs shared by the motion topics: synthetic sequences with exact ground truth (always there)
// and the course's test videos from data/video when present.
//
// Inputs::images are the frames (8-bit BGR, longest side ≤ 320 px for the bundled videos).
// Inputs::truth, when known, has one CV_8U image per frame:
//   background samples: the visible moving pixels (255);
//   tracking samples:   the object's whole box filled with 255 (also while it is hidden).

#include <vector>

#include <QRectF>

#include <opencv2/core.hpp>

#include "app/module.hpp"

namespace vl::app::motion_samples {

std::vector<Sample> background();
std::vector<Sample> tracking();

// The box a tracker should start from when the user has not drawn one: the true first box of a
// synthetic sample, a box on the target of a known clip, or a central box otherwise.
struct StartBox {
    QRectF box;
    bool known = false;  // false: a guess (the user should draw one)
};
StartBox defaultBox(const Inputs& in);

// Ground-truth box of frame t for tracking samples (empty when unknown).
cv::Rect2d truthBox(const Inputs& in, size_t t);

// A frame number to show, a few per second of video; used to send partial outputs regularly.
inline bool partialDue(size_t t) { return t % 6 == 5; }

}  // namespace vl::app::motion_samples
