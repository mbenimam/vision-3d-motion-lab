// Background subtraction: mean, approximate median, Σ-Δ and Gaussian models (see background.hpp
// for the equations and the steps). Course: "3D Vision and Motion", part 9.

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

// Channel c of an 8-bit grey or BGR image at (y, x).
uchar channelValue(const cv::Mat& img, int y, int x, int c) {
    return img.channels() == 1 ? img.at<uchar>(y, x) : img.at<cv::Vec3b>(y, x)[c];
}

void setChannelValue(cv::Mat& img, int y, int x, int c, uchar v) {
    if (img.channels() == 1)
        img.at<uchar>(y, x) = v;
    else
        img.at<cv::Vec3b>(y, x)[c] = v;
}

// The sign of v: −1, 0 or +1. Σ-Δ moves its estimates by this much, never by the size of the error.
float sgn(float v) { return float((v > 0) - (v < 0)); }

// Step 1: one float plane I_c per colour channel (one plane for a grey frame).
std::vector<cv::Mat> channelPlanes(const cv::Mat& frame) {
    CV_Assert(frame.depth() == CV_8U && (frame.channels() == 1 || frame.channels() == 3));
    std::vector<cv::Mat> planes;
    for (int c = 0; c < frame.channels(); ++c) {
        cv::Mat I(frame.size(), CV_32F);
        for (int y = 0; y < frame.rows; ++y)
            for (int x = 0; x < frame.cols; ++x) I.at<float>(y, x) = channelValue(frame, y, x, c);
        planes.push_back(I);
    }
    return planes;
}

// Step 2, Mean: classify against B_{t−1}, then B_t = B_{t−1} + α (I_t − B_{t−1}).
void stepMean(const cv::Mat& I, cv::Mat& B, float alpha, float tau, cv::Mat& D, cv::Mat& T) {
    for (int y = 0; y < I.rows; ++y)
        for (int x = 0; x < I.cols; ++x) {
            const float d = I.at<float>(y, x) - B.at<float>(y, x);
            D.at<float>(y, x) = std::abs(d);
            T.at<float>(y, x) = tau;
            B.at<float>(y, x) += alpha * d;
        }
}

// Step 2, Median: B_t = B_{t−1} + sgn(I_t − B_{t−1}). Moving by ±1 whatever the error, B converges
// to the value I_t is above half of the time: the temporal median.
void stepMedian(const cv::Mat& I, cv::Mat& B, float tau, cv::Mat& D, cv::Mat& T) {
    for (int y = 0; y < I.rows; ++y)
        for (int x = 0; x < I.cols; ++x) {
            float& b = B.at<float>(y, x);
            b += sgn(I.at<float>(y, x) - b);
            D.at<float>(y, x) = std::abs(I.at<float>(y, x) - b);
            T.at<float>(y, x) = tau;
        }
}

// Step 2, Σ-Δ (Manzanera & Richefeu 2007):
//   B_t = B_{t−1} + sgn(I_t − B_{t−1})
//   O_t = |I_t − B_t|
//   V_t = V_{t−1} + sgn(N·O_t − V_{t−1})   when O_t ≠ 0, every few frames; V kept in [Vmin, 255]
//   T   = V_t
// V is a second Σ-Δ estimate, of the dispersion: where the pixel fluctuates a lot, V rises and the
// threshold with it. Only additions, comparisons and ±1: no learning rate to tune.
void stepSigmaDelta(const cv::Mat& I, cv::Mat& B, cv::Mat& V, const BgParams& p, bool updateV, cv::Mat& D,
                    cv::Mat& T) {
    const float N = float(p.amplification), vmin = float(p.minVariance);
    for (int y = 0; y < I.rows; ++y)
        for (int x = 0; x < I.cols; ++x) {
            float& b = B.at<float>(y, x);
            float& v = V.at<float>(y, x);
            b += sgn(I.at<float>(y, x) - b);
            const float O = std::abs(I.at<float>(y, x) - b);
            if (updateV && O != 0) v = std::clamp(v + sgn(std::min(255.f, N * O) - v), vmin, 255.f);
            D.at<float>(y, x) = O;
            T.at<float>(y, x) = v;
        }
}

// Step 2, Gaussian: D = |I_t − B_{t−1}| against T = N·σ, σ = √V. The model learns in step 4.
void gaussianDifference(const cv::Mat& I, const cv::Mat& B, const cv::Mat& V, float N, cv::Mat& D, cv::Mat& T) {
    for (int y = 0; y < I.rows; ++y)
        for (int x = 0; x < I.cols; ++x) {
            D.at<float>(y, x) = std::abs(I.at<float>(y, x) - B.at<float>(y, x));
            T.at<float>(y, x) = N * std::sqrt(V.at<float>(y, x));
        }
}

// Step 3: average D and T over the channels (an L1 colour distance against the mean threshold:
// averaging the channels' noise gives fewer false alarms than an OR of per-channel tests), then
// E = (D > T).
BgStep classify(const std::vector<cv::Mat>& D, const std::vector<cv::Mat>& T) {
    const cv::Size size = D.front().size();
    const float C = float(D.size());
    BgStep s;
    s.difference.create(size, CV_32F);
    s.threshold.create(size, CV_32F);
    s.foreground.create(size, CV_8U);
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x) {
            float d = 0, t = 0;
            for (size_t c = 0; c < D.size(); ++c) {
                d += D[c].at<float>(y, x);
                t += T[c].at<float>(y, x);
            }
            s.difference.at<float>(y, x) = d / C;
            s.threshold.at<float>(y, x) = t / C;
            s.foreground.at<uchar>(y, x) = d > t ? 255 : 0;
        }
    return s;
}

// Step 4, Gaussian, after classifying (D_t = I_t − B_{t−1}):
//   B_t = B_{t−1} + α_t D_t,   V_t = V_{t−1} + α_t (D_t² − V_{t−1}),   α_t = α if E_t = 0, α/10 if E_t = 1
// The bi-level rate keeps a passing object from inflating the variance: otherwise the threshold
// would grow behind it and the next object would be missed. V is floored at 1 (σ ≥ 1 grey level).
constexpr float kForegroundRate = 0.1f;

void updateGaussian(const cv::Mat& I, cv::Mat& B, cv::Mat& V, const cv::Mat& E, float alpha) {
    for (int y = 0; y < I.rows; ++y)
        for (int x = 0; x < I.cols; ++x) {
            const float rate = E.at<uchar>(y, x) ? alpha * kForegroundRate : alpha;
            const float d = I.at<float>(y, x) - B.at<float>(y, x);
            B.at<float>(y, x) += rate * d;
            V.at<float>(y, x) = std::max(1.f, V.at<float>(y, x) + rate * (d * d - V.at<float>(y, x)));
        }
}

}  // namespace

BackgroundModel::BackgroundModel(const BgParams& p, const cv::Mat& background, double noise) : p_(p) {
    CV_Assert(p.alpha > 0.f && p.alpha <= 1.f && p.amplification >= 1);
    B_ = channelPlanes(background);
    const double sigma = std::max(1.0, noise);
    // V starts from the noise level: N·σ for Σ-Δ (an integer, it moves by ±1), σ² for the Gaussian.
    double v0 = 0;
    if (p_.method == BgMethod::SigmaDelta)
        v0 = std::round(std::clamp(p_.amplification * sigma, double(p_.minVariance), 255.0));
    if (p_.method == BgMethod::Gaussian) v0 = sigma * sigma;
    for (size_t c = 0; c < B_.size(); ++c) V_.push_back(cv::Mat(background.size(), CV_32F, cv::Scalar(v0)));
}

BgStep BackgroundModel::apply(const cv::Mat& frame) {
    CV_Assert(frame.channels() == int(B_.size()) && frame.size() == B_.front().size());
    // 1. One plane per channel.
    const std::vector<cv::Mat> I = channelPlanes(frame);
    const bool updateV = t_ % std::max(1, p_.varianceEvery) == 0;

    // 2. Per channel: difference, threshold and (except Gaussian) the update of the model.
    std::vector<cv::Mat> D, T;
    for (size_t c = 0; c < I.size(); ++c) {
        D.emplace_back(frame.size(), CV_32F);
        T.emplace_back(frame.size(), CV_32F);
        switch (p_.method) {
        case BgMethod::Mean: stepMean(I[c], B_[c], p_.alpha, p_.tau, D[c], T[c]); break;
        case BgMethod::Median: stepMedian(I[c], B_[c], p_.tau, D[c], T[c]); break;
        case BgMethod::SigmaDelta: stepSigmaDelta(I[c], B_[c], V_[c], p_, updateV, D[c], T[c]); break;
        case BgMethod::Gaussian: gaussianDifference(I[c], B_[c], V_[c], float(p_.amplification), D[c], T[c]); break;
        }
    }

    // 3. Foreground where the mean difference exceeds the mean threshold.
    BgStep s = classify(D, T);

    // 4. The Gaussian model learns now, knowing which pixels were foreground.
    if (p_.method == BgMethod::Gaussian)
        for (size_t c = 0; c < I.size(); ++c) updateGaussian(I[c], B_[c], V_[c], s.foreground, p_.alpha);
    ++t_;
    return s;
}

cv::Mat BackgroundModel::background() const {
    const int C = int(B_.size());
    cv::Mat out(B_.front().size(), CV_8UC(C));
    for (int y = 0; y < out.rows; ++y)
        for (int x = 0; x < out.cols; ++x)
            for (int c = 0; c < C; ++c)
                setChannelValue(out, y, x, c, cv::saturate_cast<uchar>(B_[size_t(c)].at<float>(y, x)));
    return out;
}

cv::Mat medianBackground(const std::vector<cv::Mat>& frames, int samples) {
    CV_Assert(!frames.empty() && samples >= 1);
    // K frames spread evenly from the first to the last.
    const int K = std::min<int>(samples, int(frames.size()));
    std::vector<const cv::Mat*> picks;
    for (int k = 0; k < K; ++k) {
        const size_t i = K == 1 ? 0 : size_t(std::lround(double(k) * double(frames.size() - 1) / double(K - 1)));
        CV_Assert(frames[i].type() == frames[0].type() && frames[i].size() == frames[0].size());
        picks.push_back(&frames[i]);
    }
    cv::Mat out(frames[0].size(), frames[0].type());
    std::vector<uchar> values(static_cast<size_t>(K));
    for (int y = 0; y < out.rows; ++y)
        for (int x = 0; x < out.cols; ++x)
            for (int c = 0; c < out.channels(); ++c) {
                for (int k = 0; k < K; ++k) values[size_t(k)] = channelValue(*picks[size_t(k)], y, x, c);
                // The middle value without sorting everything: O(K) on average.
                std::nth_element(values.begin(), values.begin() + K / 2, values.end());
                setChannelValue(out, y, x, c, values[size_t(K / 2)]);
            }
    return out;
}

double noiseSigma(const cv::Mat& frame, const cv::Mat& background) {
    CV_Assert(frame.type() == background.type() && frame.size() == background.size() && frame.depth() == CV_8U);
    // The median of 8-bit values from their histogram: exact, and O(W·H·C).
    long hist[256] = {};
    for (int y = 0; y < frame.rows; ++y)
        for (int x = 0; x < frame.cols; ++x)
            for (int c = 0; c < frame.channels(); ++c)
                ++hist[std::abs(int(channelValue(frame, y, x, c)) - int(channelValue(background, y, x, c)))];
    const long half = long(frame.total()) * frame.channels() / 2;
    long below = 0;
    int median = 0;
    while (median < 255 && (below += hist[median]) <= half) ++median;
    // For Gaussian noise, median |X − μ| = 0.6745 σ, hence σ = 1.4826 · median.
    return 1.4826 * median;
}

Confusion compareMasks(const cv::Mat& detected, const cv::Mat& truth) {
    CV_Assert(detected.type() == CV_8U && truth.type() == CV_8U && detected.size() == truth.size());
    Confusion c;
    for (int y = 0; y < truth.rows; ++y)
        for (int x = 0; x < truth.cols; ++x) {
            const bool d = detected.at<uchar>(y, x) != 0, t = truth.at<uchar>(y, x) != 0;
            c.tp += d && t;
            c.fp += d && !t;
            c.fn += !d && t;
            c.tn += !d && !t;
        }
    return c;
}

}  // namespace vl::motion
