#include "core/flow/opencv_baselines.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "core/common/cv_compat.hpp"
#include "core/common/task.hpp"

namespace vl::flow::opencv {
namespace {

cv::Mat grey(const cv::Mat& img) {
    if (img.channels() == 1) return img;
    cv::Mat g;
    cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    return g;
}

}  // namespace

Flow pyramidalLucasKanade(const cv::Mat& frame0, const cv::Mat& frame1, int windowRadius, int levels) {
    const cv::Mat g0 = grey(frame0), g1 = grey(frame1);
    std::vector<cv::Point2f> from;
    from.reserve(g0.total());
    for (int y = 0; y < g0.rows; ++y)
        for (int x = 0; x < g0.cols; ++x) from.emplace_back(float(x), float(y));
    std::vector<cv::Point2f> to;
    std::vector<uchar> status;
    std::vector<float> error;
    const int side = 2 * std::max(1, windowRadius) + 1;
    const Stopwatch clock;
    cv::calcOpticalFlowPyrLK(g0, g1, from, to, status, error, cv::Size(side, side), std::max(0, levels - 1));
    Flow out{cv::Mat(g0.size(), CV_32FC2), clock.ms()};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (size_t i = 0; i < from.size(); ++i) {
        const cv::Point2f d = to[i] - from[i];
        out.flow.at<cv::Vec2f>(int(from[i].y), int(from[i].x)) = status[i] ? cv::Vec2f(d.x, d.y) : cv::Vec2f(nan, nan);
    }
    return out;
}

Flow farneback(const cv::Mat& frame0, const cv::Mat& frame1, int levels) {
    const cv::Mat g0 = grey(frame0), g1 = grey(frame1);
    Flow out;
    const Stopwatch clock;
    cv::calcOpticalFlowFarneback(g0, g1, out.flow, 0.5, std::max(1, levels), 15, 3, 5, 1.2, 0);
    out.ms = clock.ms();
    return out;
}

Flow dis(const cv::Mat& frame0, const cv::Mat& frame1) {
    const cv::Mat g0 = grey(frame0), g1 = grey(frame1);
    Flow out;
    auto method = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
    const Stopwatch clock;
    method->calc(g0, g1, out.flow);
    out.ms = clock.ms();
    return out;
}

}  // namespace vl::flow::opencv
