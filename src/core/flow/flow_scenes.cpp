#include "core/flow/flow_scenes.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace vl::flow {

namespace {

// Share of pixel [x, x+1) x [y, y+1) covered by the rectangle r.
double coverage(const cv::Rect2d& r, int x, int y) {
    const double ox = std::max(0.0, std::min(r.x + r.width, x + 1.0) - std::max(r.x, double(x)));
    const double oy = std::max(0.0, std::min(r.y + r.height, y + 1.0) - std::max(r.y, double(y)));
    return ox * oy;
}

cv::Mat render(const cv::Mat& background, const cv::Rect2d& r, const cv::Vec3f& colour) {
    cv::Mat out = background.clone();
    for (int y = std::max(0, int(r.y) - 1); y < std::min(out.rows, int(r.y + r.height) + 2); ++y) {
        auto* o = out.ptr<cv::Vec3b>(y);
        for (int x = std::max(0, int(r.x) - 1); x < std::min(out.cols, int(r.x + r.width) + 2); ++x) {
            const double a = coverage(r, x, y);
            for (int c = 0; c < 3; ++c) o[x][c] = cv::saturate_cast<uchar>((1 - a) * o[x][c] + a * colour[c]);
        }
    }
    return out;
}

}  // namespace

vl::synth::FlowPair slidingSquare(cv::Size size, cv::Rect2d square, cv::Point2d shift, std::uint32_t seed) {
    vl::synth::FlowPair p;
    // Background texture at a third of its contrast and blurred (σ 3 px): the square's sharp edges
    // dominate the windows they cross (a clean aperture), while the background stays measurable.
    cv::Mat bg;
    cv::GaussianBlur(vl::synth::colorTexture(size, seed), bg, cv::Size(), 3.0);
    bg.convertTo(bg, CV_8UC3, 0.35, 0.65 * 110);
    const cv::Vec3f colour(60, 150, 235);  // flat orange (BGR)
    p.frame0 = render(bg, square, colour);
    p.frame1 = render(bg, cv::Rect2d(square.x + shift.x, square.y + shift.y, square.width, square.height), colour);
    p.flow = cv::Mat::zeros(size, CV_32FC2);
    for (int y = 0; y < size.height; ++y) {
        auto* f = p.flow.ptr<cv::Vec2f>(y);
        for (int x = 0; x < size.width; ++x)
            if (coverage(square, x, y) >= 0.5) f[x] = {float(shift.x), float(shift.y)};
    }
    return p;
}

}  // namespace vl::flow
