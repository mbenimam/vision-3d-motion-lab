#include "core/motion/tracking.hpp"

namespace vl::motion {

double iou(const cv::Rect2d& a, const cv::Rect2d& b) {
    const double inter = (a & b).area();
    const double uni = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.0;
}

}  // namespace vl::motion
