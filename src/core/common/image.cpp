#include "core/common/image.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace vl {

cv::Mat toGrayF(const cv::Mat& img) {
    CV_Assert(!img.empty());
    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4)
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    else
        gray = img;
    cv::Mat out;
    gray.convertTo(out, CV_32F);
    return out;
}

cv::Mat limitSize(const cv::Mat& img, int maxSide, double* scale) {
    const int side = std::max(img.cols, img.rows);
    if (side <= maxSide || side == 0) {
        if (scale) *scale = 1.0;
        return img;
    }
    const double s = static_cast<double>(maxSide) / side;
    if (scale) *scale = s;
    cv::Mat out;
    cv::resize(img, out, cv::Size(), s, s, cv::INTER_AREA);
    return out;
}

cv::Mat colorize(const cv::Mat& values, double lo, double hi, const cv::Mat& valid) {
    CV_Assert(values.channels() == 1);
    cv::Mat v8;
    const double range = hi > lo ? hi - lo : 1.0;
    values.convertTo(v8, CV_8U, 255.0 / range, -lo * 255.0 / range);
    cv::Mat out;
    cv::applyColorMap(v8, out, cv::COLORMAP_TURBO);
    if (!valid.empty()) out.setTo(cv::Scalar::all(0), valid == 0);
    return out;
}

cv::Mat toBGR(const cv::Mat& img) {
    cv::Mat src = img;
    if (img.depth() != CV_8U) img.convertTo(src, CV_8U);
    if (src.channels() == 3) return src;
    cv::Mat out;
    cv::cvtColor(src, out, src.channels() == 4 ? cv::COLOR_BGRA2BGR : cv::COLOR_GRAY2BGR);
    return out;
}

}  // namespace vl
