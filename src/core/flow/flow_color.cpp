#include "core/flow/flow_color.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace vl::flow {

namespace {

// The 55-colour Middlebury wheel (RGB): red -> yellow -> green -> cyan -> blue -> magenta -> red,
// with more steps where the eye tells hues apart better (Baker et al., colorcode.cpp).
const std::array<cv::Vec3f, 55>& wheel() {
    static const auto w = [] {
        std::array<cv::Vec3f, 55> c{};
        int k = 0;
        constexpr int RY = 15, YG = 6, GC = 4, CB = 11, BM = 13, MR = 6;
        for (int i = 0; i < RY; ++i) c[size_t(k++)] = {255.f, 255.f * float(i) / RY, 0.f};
        for (int i = 0; i < YG; ++i) c[size_t(k++)] = {255.f - 255.f * float(i) / YG, 255.f, 0.f};
        for (int i = 0; i < GC; ++i) c[size_t(k++)] = {0.f, 255.f, 255.f * float(i) / GC};
        for (int i = 0; i < CB; ++i) c[size_t(k++)] = {0.f, 255.f - 255.f * float(i) / CB, 255.f};
        for (int i = 0; i < BM; ++i) c[size_t(k++)] = {255.f * float(i) / BM, 0.f, 255.f};
        for (int i = 0; i < MR; ++i) c[size_t(k++)] = {255.f, 0.f, 255.f - 255.f * float(i) / MR};
        return c;
    }();
    return w;
}

float percentile(std::vector<float>& v, double q) {
    if (v.empty()) return 0.f;
    const auto k = static_cast<std::ptrdiff_t>(q * double(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[size_t(k)];
}

std::vector<float> magnitudes(const cv::Mat& flow) {
    std::vector<float> m;
    m.reserve(flow.total());
    for (int y = 0; y < flow.rows; ++y) {
        const auto* f = flow.ptr<cv::Vec2f>(y);
        for (int x = 0; x < flow.cols; ++x)
            if (std::isfinite(f[x][0]) && std::isfinite(f[x][1]) && std::abs(f[x][0]) < 1e9f && std::abs(f[x][1]) < 1e9f)
                m.push_back(std::hypot(f[x][0], f[x][1]));
    }
    return m;
}

}  // namespace

cv::Vec3b flowColor(float u, float v, float maxMagnitude) {
    if (!std::isfinite(u) || !std::isfinite(v)) return {0, 0, 0};
    const auto& c = wheel();
    const float n = float(c.size());
    const float fu = u / maxMagnitude, fv = v / maxMagnitude;
    const float rad = std::sqrt(fu * fu + fv * fv);
    const float a = std::atan2(-fv, -fu) / std::numbers::pi_v<float>;  // in [-1, 1]
    const float fk = (a + 1.f) / 2.f * (n - 1.f);                       // position on the wheel
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % int(n);
    const float t = fk - float(k0);
    cv::Vec3b out;
    for (int ch = 0; ch < 3; ++ch) {
        float col = ((1.f - t) * c[size_t(k0)][ch] + t * c[size_t(k1)][ch]) / 255.f;
        // Inside the unit disc: fade to white at the centre. Outside: keep the hue, darken.
        col = rad <= 1.f ? 1.f - rad * (1.f - col) : col * 0.75f;
        out[2 - ch] = cv::saturate_cast<uchar>(255.f * col);  // RGB wheel -> BGR pixel
    }
    return out;
}

cv::Mat flowToColor(const cv::Mat& flow, float maxMagnitude) {
    CV_Assert(flow.type() == CV_32FC2);
    const float m = std::max(maxMagnitude, 1e-6f);
    cv::Mat out(flow.size(), CV_8UC3);
    for (int y = 0; y < flow.rows; ++y) {
        const auto* f = flow.ptr<cv::Vec2f>(y);
        auto* o = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < flow.cols; ++x) o[x] = flowColor(f[x][0], f[x][1], m);
    }
    return out;
}

cv::Point2f wheelPosition(cv::Size size, float maxMagnitude, cv::Point2f vec, int margin) {
    const float r = float(std::min(size.width, size.height)) / 2.f - float(margin);
    const cv::Point2f c(float(size.width) / 2.f, float(size.height) / 2.f);
    return c + vec * (r / std::max(maxMagnitude, 1e-6f));
}

cv::Mat colorWheel(cv::Size size, float maxMagnitude, int margin) {
    cv::Mat out(size, CV_8UC3, cv::Scalar(30, 26, 24));
    const float r = float(std::min(size.width, size.height)) / 2.f - float(margin);
    const float cx = float(size.width) / 2.f, cy = float(size.height) / 2.f;
    for (int y = 0; y < size.height; ++y) {
        auto* o = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < size.width; ++x) {
            const float dx = (float(x) + 0.5f - cx) / r, dy = (float(y) + 0.5f - cy) / r;
            if (dx * dx + dy * dy <= 1.f) o[x] = flowColor(dx * maxMagnitude, dy * maxMagnitude, maxMagnitude);
        }
    }
    return out;
}

float robustMaxMagnitude(const cv::Mat& flow) {
    auto m = magnitudes(flow);
    return std::max(1.f, percentile(m, 0.99));
}

std::vector<Arrow> flowArrows(const cv::Mat& flow, int step, float scale, float minLength) {
    CV_Assert(flow.type() == CV_32FC2 && step > 0);
    std::vector<Arrow> out;
    for (int y = step / 2; y < flow.rows; y += step) {
        const auto* f = flow.ptr<cv::Vec2f>(y);
        for (int x = step / 2; x < flow.cols; x += step) {
            const float u = f[x][0], v = f[x][1];
            if (!std::isfinite(u) || !std::isfinite(v) || std::hypot(u, v) < minLength) continue;
            const cv::Point2f p(float(x) + 0.5f, float(y) + 0.5f);
            out.push_back({p, p + cv::Point2f(u, v) * scale});
        }
    }
    return out;
}

float arrowScale(const cv::Mat& flow, int step) {
    auto m = magnitudes(flow);
    const float p95 = percentile(m, 0.95);
    return p95 > 1e-3f ? std::max(1.f, 0.9f * float(step) / p95) : 1.f;
}

}  // namespace vl::flow
