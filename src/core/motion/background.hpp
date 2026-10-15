#pragma once
// Background subtraction with a static camera. Course: "3D Vision and Motion", part 9.
//
// Every pixel keeps an estimate B of the empty scene, updated online with each frame I_t, and is
// labelled foreground (E_t = 1) where the frame is too far from it, |I_t − B| > T. The four models
// differ in how B follows I and where the threshold T comes from:
//
//   Mean        recursive average     B_t = B_{t−1} + α (I_t − B_{t−1});          T = τ (fixed)
//   Median      approximate median    B_t = B_{t−1} + sgn(I_t − B_{t−1});         T = τ (fixed)
//   Σ-Δ         Manzanera's sigma-delta: B by ±1 as above, O_t = |I_t − B_t|, then a dispersion
//               V_t = V_{t−1} + sgn(N·O_t − V_{t−1}) every few frames (slower than B);  T = V_t
//   Gaussian    running mean and variance, with D_t = I_t − B_{t−1}:
//               B_t = B_{t−1} + α_t D_t,  V_t = V_{t−1} + α_t (D_t² − V_{t−1});  T = N·√V
//               α_t = α on background pixels, α/10 on foreground ones
//
// Colour frames run one model per channel (B, G, R); a pixel is foreground when the channels' mean
// difference exceeds their mean threshold (grey levels alone confuse an orange object with a
// background of the same brightness).
//
// One frame, in BackgroundModel::apply:
//   1. channelPlanes        split the frame into one float plane I_c per colour channel
//   2. stepMean / stepMedian / stepSigmaDelta / gaussianDifference
//                           per channel: the difference D_c = |I_c − B_c| and threshold T_c
//                           (and, except for the Gaussian, the update of B_c and V_c)
//   3. classify             average D and T over the channels, E = (D > T)
//   4. updateGaussian       the Gaussian model learns after classifying, at a rate that depends on E
//
// Every update is a constant number of operations per pixel and channel: O(W·H·C) per frame, and no
// history is kept. The start B_0 can be the median of a few frames spread over the clip
// (medianBackground), so objects present in the first frame do not leave a ghost.

#include <vector>

#include <opencv2/core.hpp>

namespace vl::motion {

enum class BgMethod { Mean, Median, SigmaDelta, Gaussian };

struct BgParams {
    BgMethod method = BgMethod::SigmaDelta;
    float alpha = 0.02f;    // α: learning rate of Mean and Gaussian, in (0, 1]
    float tau = 25.f;       // τ: fixed threshold of Mean and Median, in grey levels
    int amplification = 3;  // N: Σ-Δ makes V follow N·O, Gaussian thresholds at N·σ
    int varianceEvery = 2;  // Σ-Δ updates V once every this many frames (V moves slower than B)
    int minVariance = 2;    // Σ-Δ keeps V in [minVariance, 255]
};

// What one frame produced. All images are single-channel with the frame's size; for colour frames,
// difference and threshold are averaged over the channels, and foreground == (difference > threshold).
struct BgStep {
    cv::Mat difference;  // CV_32F, |I_t − B| in grey levels
    cv::Mat threshold;   // CV_32F, the per-pixel threshold T it was compared to
    cv::Mat foreground;  // CV_8U, 255 where difference > threshold (E_t)
};

class BackgroundModel {
public:
    // background: 8-bit grey or BGR start B_0 (e.g. the first frame, or medianBackground()).
    // noise: a guess of the sensor noise σ in grey levels, used to start the dispersion V of Σ-Δ
    // and Gaussian (they would otherwise flag every pixel until V has grown).
    BackgroundModel(const BgParams& p, const cv::Mat& background, double noise = 2.0);

    // Classify a frame (same type as the start background) and update the model with it.
    BgStep apply(const cv::Mat& frame);

    cv::Mat background() const;  // the current B, 8-bit, same type as the frames
    const BgParams& params() const { return p_; }
    long frames() const { return t_; }

private:
    BgParams p_;
    std::vector<cv::Mat> B_;  // per channel, CV_32F: the background estimate
    std::vector<cv::Mat> V_;  // per channel, CV_32F: Σ-Δ dispersion, or Gaussian variance σ²
    long t_ = 0;
};

// Per-pixel (and per-channel) median of `samples` frames spread evenly over the clip. A moving
// object covers any pixel in only a few of them, so the median is the empty scene. O(K·W·H·C).
cv::Mat medianBackground(const std::vector<cv::Mat>& frames, int samples = 15);

// Robust sensor-noise estimate: 1.4826 · median |I − B| over all pixels and channels (grey levels).
double noiseSigma(const cv::Mat& frame, const cv::Mat& background);

// 8-bit grey copy of an 8-bit grey or BGR frame.
cv::Mat toGray8(const cv::Mat& img);

// Pixel counts of a detection against a ground-truth mask (both CV_8U, non-zero = foreground).
struct Confusion {
    long tp = 0, fp = 0, fn = 0, tn = 0;
    double precision() const { return tp + fp ? double(tp) / double(tp + fp) : 1.0; }
    double recall() const { return tp + fn ? double(tp) / double(tp + fn) : 1.0; }
    // F1 = 2 TP / (2 TP + FP + FN), the harmonic mean of precision and recall.
    double f1() const {
        const double d = 2.0 * double(tp) + double(fp) + double(fn);
        return d > 0 ? 2.0 * double(tp) / d : 1.0;
    }
};
Confusion compareMasks(const cv::Mat& detected, const cv::Mat& truth);

}  // namespace vl::motion
