// Measuring a flow against ground truth, and the Middlebury .flo format. Course: "3D Vision and
// Motion", part 11 (quantitative evaluation of optical flow, Baker et al. 2011).
//
//   endpoint error  EPE(x) = |w(x) − w_gt(x)| = √((u − u_gt)² + (v − v_gt)²)
//   angular error   AE(x)  = arccos( (u·u_gt + v·v_gt + 1) / (√(u² + v² + 1) · √(u_gt² + v_gt² + 1)) )
//   residual        |I1(x + w(x)) − I0(x)|, what the flow leaves unexplained (needs no truth)

#include "core/flow/flow_eval.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numbers>
#include <stdexcept>

#include "core/common/image.hpp"
#include "core/flow/flow_common.hpp"

namespace vl::flow {

namespace {

constexpr float kTag = 202021.25f;  // "PIEH" read as a little-endian float
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

bool finite(const cv::Vec2f& w) { return std::isfinite(w[0]) && std::isfinite(w[1]); }

double endpointError(const cv::Vec2f& w, const cv::Vec2f& t) { return std::hypot(double(w[0]) - t[0], double(w[1]) - t[1]); }

// The angle between the space-time vectors (u, v, 1) and (u_gt, v_gt, 1), in degrees.
double angularError(const cv::Vec2f& w, const cv::Vec2f& t) {
    const double dot = double(w[0]) * t[0] + double(w[1]) * t[1] + 1.0;
    const double lengths = std::sqrt((double(w[0]) * w[0] + double(w[1]) * w[1] + 1.0) * (double(t[0]) * t[0] + double(t[1]) * t[1] + 1.0));
    return std::acos(std::clamp(dot / lengths, -1.0, 1.0)) * 180.0 / std::numbers::pi;
}

// One error per pixel, NaN where the truth is unknown or the estimate missing.
cv::Mat errorMap(const cv::Mat& flow, const cv::Mat& truth, double (*error)(const cv::Vec2f&, const cv::Vec2f&)) {
    CV_Assert(flow.type() == CV_32FC2 && truth.type() == CV_32FC2 && flow.size() == truth.size());
    cv::Mat out(flow.size(), CV_32F);
    for (int y = 0; y < flow.rows; ++y)
        for (int x = 0; x < flow.cols; ++x) {
            const cv::Vec2f w = flow.at<cv::Vec2f>(y, x), t = truth.at<cv::Vec2f>(y, x);
            out.at<float>(y, x) = isKnownFlow(t) && finite(w) ? float(error(w, t)) : kNaN;
        }
    return out;
}

}  // namespace

bool isKnownFlow(const cv::Vec2f& w) { return finite(w) && std::abs(w[0]) <= 1e9f && std::abs(w[1]) <= 1e9f; }

FlowErrors evaluateFlow(const cv::Mat& flow, const cv::Mat& truth) {
    CV_Assert(flow.type() == CV_32FC2 && truth.type() == CV_32FC2 && flow.size() == truth.size());
    FlowErrors e;
    double sumEpe = 0, sumAe = 0;
    long over1 = 0, over3 = 0;
    for (int y = 0; y < flow.rows; ++y)
        for (int x = 0; x < flow.cols; ++x) {
            const cv::Vec2f w = flow.at<cv::Vec2f>(y, x), t = truth.at<cv::Vec2f>(y, x);
            if (!isKnownFlow(t)) continue;
            if (!finite(w)) {
                ++e.missing;
                continue;
            }
            const double epe = endpointError(w, t);
            sumEpe += epe;
            sumAe += angularError(w, t);
            over1 += epe > 1.0;
            over3 += epe > 3.0;
            ++e.count;
        }
    if (e.count) {
        const double n = double(e.count);
        e.epe = sumEpe / n;
        e.angular = sumAe / n;
        e.over1 = double(over1) / n;
        e.over3 = double(over3) / n;
    }
    return e;
}

cv::Mat endpointErrorMap(const cv::Mat& flow, const cv::Mat& truth) { return errorMap(flow, truth, endpointError); }

cv::Mat angularErrorMap(const cv::Mat& flow, const cv::Mat& truth) { return errorMap(flow, truth, angularError); }

cv::Mat brightnessResidual(const cv::Mat& frame0, const cv::Mat& frame1, const cv::Mat& flow) {
    const cv::Mat I0 = vl::toGrayF(frame0), I1 = vl::toGrayF(frame1);
    cv::Mat outside;
    const cv::Mat I1w = warpImage(I1, flow, &outside);
    cv::Mat r(I0.size(), CV_32F);
    for (int y = 0; y < r.rows; ++y)
        for (int x = 0; x < r.cols; ++x) {
            const bool known = finite(flow.at<cv::Vec2f>(y, x)) && outside.at<uchar>(y, x) == 0;
            r.at<float>(y, x) = known ? std::abs(I1w.at<float>(y, x) - I0.at<float>(y, x)) : kNaN;
        }
    return r;
}

cv::Mat readFlo(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    float tag = 0;
    std::int32_t w = 0, h = 0;
    in.read(reinterpret_cast<char*>(&tag), 4);
    in.read(reinterpret_cast<char*>(&w), 4);
    in.read(reinterpret_cast<char*>(&h), 4);
    if (!in || tag != kTag) throw std::runtime_error(path + ": not a .flo file (bad tag)");
    if (w <= 0 || h <= 0 || w > 100000 || h > 100000) throw std::runtime_error(path + ": bad size");
    cv::Mat flow(h, w, CV_32FC2);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) in.read(reinterpret_cast<char*>(&flow.at<cv::Vec2f>(y, x)), 2 * sizeof(float));
    if (!in) throw std::runtime_error(path + ": truncated");
    return flow;
}

void writeFlo(const std::string& path, const cv::Mat& flow) {
    CV_Assert(flow.type() == CV_32FC2);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    const std::int32_t w = flow.cols, h = flow.rows;
    out.write(reinterpret_cast<const char*>(&kTag), 4);
    out.write(reinterpret_cast<const char*>(&w), 4);
    out.write(reinterpret_cast<const char*>(&h), 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) out.write(reinterpret_cast<const char*>(&flow.at<cv::Vec2f>(y, x)), 2 * sizeof(float));
    if (!out) throw std::runtime_error("failed writing " + path);
}

}  // namespace vl::flow
