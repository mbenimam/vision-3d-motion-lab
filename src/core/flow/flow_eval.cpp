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

double angleDeg(const cv::Vec2f& w, const cv::Vec2f& t) {
    const double num = double(w[0]) * t[0] + double(w[1]) * t[1] + 1.0;
    const double den = std::sqrt((double(w[0]) * w[0] + double(w[1]) * w[1] + 1.0) * (double(t[0]) * t[0] + double(t[1]) * t[1] + 1.0));
    return std::acos(std::clamp(num / den, -1.0, 1.0)) * 180.0 / std::numbers::pi;
}

template <class F>
cv::Mat errorMap(const cv::Mat& flow, const cv::Mat& truth, F f) {
    CV_Assert(flow.type() == CV_32FC2 && truth.type() == CV_32FC2 && flow.size() == truth.size());
    cv::Mat out(flow.size(), CV_32F);
    for (int y = 0; y < flow.rows; ++y) {
        const auto* w = flow.ptr<cv::Vec2f>(y);
        const auto* t = truth.ptr<cv::Vec2f>(y);
        float* o = out.ptr<float>(y);
        for (int x = 0; x < flow.cols; ++x) o[x] = isKnownFlow(t[x]) && finite(w[x]) ? float(f(w[x], t[x])) : kNaN;
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
    for (int y = 0; y < flow.rows; ++y) {
        const auto* w = flow.ptr<cv::Vec2f>(y);
        const auto* t = truth.ptr<cv::Vec2f>(y);
        for (int x = 0; x < flow.cols; ++x) {
            if (!isKnownFlow(t[x])) continue;
            if (!finite(w[x])) {
                ++e.missing;
                continue;
            }
            const double epe = std::hypot(double(w[x][0]) - t[x][0], double(w[x][1]) - t[x][1]);
            sumEpe += epe;
            sumAe += angleDeg(w[x], t[x]);
            over1 += epe > 1.0;
            over3 += epe > 3.0;
            ++e.count;
        }
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

cv::Mat endpointErrorMap(const cv::Mat& flow, const cv::Mat& truth) {
    return errorMap(flow, truth, [](const cv::Vec2f& w, const cv::Vec2f& t) {
        return std::hypot(double(w[0]) - t[0], double(w[1]) - t[1]);
    });
}

cv::Mat angularErrorMap(const cv::Mat& flow, const cv::Mat& truth) { return errorMap(flow, truth, angleDeg); }

cv::Mat brightnessResidual(const cv::Mat& frame0, const cv::Mat& frame1, const cv::Mat& flow) {
    const cv::Mat I0 = vl::toGrayF(frame0), I1 = vl::toGrayF(frame1);
    cv::Mat outside;
    const cv::Mat I1w = warpImage(I1, flow, &outside);
    cv::Mat r = cv::abs(I1w - I0);
    r.setTo(kNaN, outside);
    for (int y = 0; y < flow.rows; ++y) {
        const auto* w = flow.ptr<cv::Vec2f>(y);
        float* o = r.ptr<float>(y);
        for (int x = 0; x < flow.cols; ++x)
            if (!finite(w[x])) o[x] = kNaN;
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
    for (int y = 0; y < h; ++y) in.read(reinterpret_cast<char*>(flow.ptr<float>(y)), std::streamsize(w) * 2 * 4);
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
    for (int y = 0; y < h; ++y) out.write(reinterpret_cast<const char*>(flow.ptr<float>(y)), std::streamsize(w) * 2 * 4);
    if (!out) throw std::runtime_error("failed writing " + path);
}

}  // namespace vl::flow
