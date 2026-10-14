#include "core/motion/background.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace vl::motion {

cv::Mat toGray8(const cv::Mat& img) {
    CV_Assert(img.depth() == CV_8U && (img.channels() == 1 || img.channels() == 3));
    if (img.channels() == 1) return img.clone();
    cv::Mat g;
    cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    return g;
}

namespace {

// A continuous 8-bit image seen as H × (W·C) single-channel elements.
cv::Mat planar(const cv::Mat& img) {
    CV_Assert(img.depth() == CV_8U);
    return (img.isContinuous() ? img : img.clone()).reshape(1, img.rows);
}

// Gaussian model: learning rate on foreground pixels, relative to α.
constexpr float kForegroundRate = 0.1f;

// One ±1 step of a toward b: the Σ-Δ modulation (sign of the error, never its size).
inline uchar stepToward(uchar a, int b) { return uchar(a + (a < b) - (a > b)); }

}  // namespace

BackgroundModel::BackgroundModel(const BgParams& p, const cv::Mat& background, double noise)
    : p_(p), channels_(background.channels()) {
    CV_Assert(background.depth() == CV_8U && (channels_ == 1 || channels_ == 3));
    CV_Assert(p.alpha > 0.f && p.alpha <= 1.f && p.amplification >= 1);
    const double sigma = std::max(1.0, noise);
    const cv::Mat b = planar(background);
    switch (p_.method) {
    case BgMethod::Mean:
        b.convertTo(mF_, CV_32F);
        break;
    case BgMethod::Gaussian:
        b.convertTo(mF_, CV_32F);
        varF_ = cv::Mat(b.size(), CV_32F, cv::Scalar(sigma * sigma));
        break;
    case BgMethod::Median:
        m8_ = b.clone();
        break;
    case BgMethod::SigmaDelta:
        m8_ = b.clone();
        // V estimates N·|I − M|; start it at N times the noise level, inside [Vmin, 255].
        v8_ = cv::Mat(b.size(), CV_8U, cv::Scalar(std::clamp(p_.amplification * sigma, double(p_.minVariance), 255.0)));
        break;
    }
}

BgStep BackgroundModel::apply(const cv::Mat& frame) {
    CV_Assert(frame.depth() == CV_8U && frame.channels() == channels_);
    const cv::Mat I8 = planar(frame);
    CV_Assert(I8.size() == (m8_.empty() ? mF_.size() : m8_.size()));
    const int H = frame.rows, W = frame.cols, C = channels_, WC = W * C;
    BgStep s;
    s.difference.create(frame.size(), CV_32F);
    s.threshold.create(frame.size(), CV_32F);
    s.foreground.create(frame.size(), CV_8U);
    const bool updateV = p_.method == BgMethod::SigmaDelta && t_ % std::max(1, p_.varianceEvery) == 0;
    const int N = p_.amplification, vmin = p_.minVariance;
    const float a = p_.alpha, tau = p_.tau, n = float(N);
    std::vector<float> dRow(static_cast<size_t>(WC)), tRow(static_cast<size_t>(WC));  // per-element Δ and T of the current row
    float* D = dRow.data();
    float* T = tRow.data();

    for (int y = 0; y < H; ++y) {
        const uchar* I = I8.ptr<uchar>(y);
        switch (p_.method) {
        case BgMethod::Median: {
            uchar* M = m8_.ptr<uchar>(y);
            for (int i = 0; i < WC; ++i) {
                M[i] = stepToward(M[i], I[i]);
                D[i] = float(std::abs(int(I[i]) - int(M[i])));
                T[i] = tau;
            }
            break;
        }
        case BgMethod::SigmaDelta: {
            // Manzanera & Richefeu (2007): M follows I by ±1, V follows N·Δ by ±1 (where Δ ≠ 0),
            // foreground where Δ exceeds V. Integers only: difference, comparison, increment.
            uchar* M = m8_.ptr<uchar>(y);
            uchar* V = v8_.ptr<uchar>(y);
            for (int i = 0; i < WC; ++i) {
                M[i] = stepToward(M[i], I[i]);
                const int delta = std::abs(int(I[i]) - int(M[i]));
                if (updateV && delta != 0) V[i] = uchar(std::clamp(int(stepToward(V[i], std::min(255, N * delta))), vmin, 255));
                D[i] = float(delta);
                T[i] = float(V[i]);
            }
            break;
        }
        case BgMethod::Mean: {
            float* M = mF_.ptr<float>(y);
            for (int i = 0; i < WC; ++i) {
                const float d = float(I[i]) - M[i];
                D[i] = std::abs(d);
                T[i] = tau;
                M[i] += a * d;  // classify against M_{t−1}, then update
            }
            break;
        }
        case BgMethod::Gaussian: {
            const float* M = mF_.ptr<float>(y);
            const float* S2 = varF_.ptr<float>(y);
            for (int i = 0; i < WC; ++i) {
                D[i] = std::abs(float(I[i]) - M[i]);
                T[i] = n * std::sqrt(S2[i]);
            }
            break;
        }
        }
        // Colour: average Δ and T over the channels (an L1 colour distance against the mean
        // threshold). Averaging the channels' noise lowers the false alarms of a per-channel OR.
        float* outD = s.difference.ptr<float>(y);
        float* outT = s.threshold.ptr<float>(y);
        uchar* E = s.foreground.ptr<uchar>(y);
        const float inv = 1.f / float(C);
        for (int x = 0; x < W; ++x) {
            float d = 0, t = 0;
            for (int c = 0; c < C; ++c) {
                d += D[x * C + c];
                t += T[x * C + c];
            }
            outD[x] = d * inv;
            outT[x] = t * inv;
            E[x] = d > t ? 255 : 0;
        }
        if (p_.method == BgMethod::Gaussian) {
            // Bi-level update (course: α₁ on background, α₂ ≪ α₁ on foreground): a passing object
            // must not inflate the variance, or the threshold grows and later objects are missed.
            float* M = mF_.ptr<float>(y);
            float* S2 = varF_.ptr<float>(y);
            for (int x = 0; x < W; ++x) {
                const float r = E[x] ? a * kForegroundRate : a;
                for (int c = 0; c < C; ++c) {
                    const int i = x * C + c;
                    const float d = float(I[i]) - M[i];
                    M[i] += r * d;
                    S2[i] = std::max(1.f, S2[i] + r * (d * d - S2[i]));  // floor: σ ≥ 1 grey level
                }
            }
        }
    }
    ++t_;
    return s;
}

cv::Mat BackgroundModel::background() const {
    cv::Mat b;
    if (!m8_.empty())
        b = m8_.clone();
    else
        mF_.convertTo(b, CV_8U);
    return b.reshape(channels_, b.rows);
}

cv::Mat medianBackground(const std::vector<cv::Mat>& frames, int samples) {
    CV_Assert(!frames.empty() && samples >= 1);
    const int K = std::min<int>(samples, int(frames.size()));
    std::vector<cv::Mat> picks;
    picks.reserve(size_t(K));
    for (int k = 0; k < K; ++k) {
        const size_t i = K == 1 ? 0 : size_t(std::lround(double(k) * double(frames.size() - 1) / double(K - 1)));
        CV_Assert(frames[i].type() == frames[0].type() && frames[i].size() == frames[0].size());
        picks.push_back(planar(frames[i]));
    }
    cv::Mat out(picks.front().size(), CV_8U);
    std::vector<uchar> v(static_cast<size_t>(K));
    const auto mid = v.begin() + K / 2;
    for (int y = 0; y < out.rows; ++y) {
        uchar* o = out.ptr<uchar>(y);
        for (int i = 0; i < out.cols; ++i) {
            for (int k = 0; k < K; ++k) v[size_t(k)] = picks[size_t(k)].ptr<uchar>(y)[i];
            std::nth_element(v.begin(), mid, v.end());
            o[i] = *mid;
        }
    }
    return out.reshape(frames[0].channels(), out.rows);
}

double noiseSigma(const cv::Mat& frame, const cv::Mat& background) {
    CV_Assert(frame.type() == background.type() && frame.size() == background.size() && frame.depth() == CV_8U);
    cv::Mat d;
    cv::absdiff(planar(frame), planar(background), d);
    // Median of 8-bit values from their histogram: O(W·H·C) and exact.
    long hist[256] = {};
    for (int y = 0; y < d.rows; ++y) {
        const uchar* p = d.ptr<uchar>(y);
        for (int x = 0; x < d.cols; ++x) ++hist[p[x]];
    }
    const long half = long(d.total()) / 2;
    long acc = 0;
    int med = 0;
    while (med < 255 && (acc += hist[med]) <= half) ++med;
    return 1.4826 * med;
}

Confusion compareMasks(const cv::Mat& detected, const cv::Mat& truth) {
    CV_Assert(detected.type() == CV_8U && truth.type() == CV_8U && detected.size() == truth.size());
    Confusion c;
    for (int y = 0; y < truth.rows; ++y) {
        const uchar* d = detected.ptr<uchar>(y);
        const uchar* t = truth.ptr<uchar>(y);
        for (int x = 0; x < truth.cols; ++x) {
            const bool dd = d[x] != 0, tt = t[x] != 0;
            c.tp += dd && tt;
            c.fp += dd && !tt;
            c.fn += !dd && tt;
            c.tn += !dd && !tt;
        }
    }
    return c;
}

}  // namespace vl::motion
