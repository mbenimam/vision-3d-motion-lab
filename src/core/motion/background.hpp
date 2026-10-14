#pragma once
// Per-pixel background models for a static camera, all updated online (one pass per frame).
//
// Each model keeps a background estimate M_t and decides "foreground" where the current grey level
// is too far from it: |I_t − M_t| > T_t. The models differ in how M follows I and in where the
// threshold T comes from:
//
//   Mean        exponential average  M ← M + α (I − M); fixed threshold τ.
//   Median      approximate median   M ← M ± 1 toward I; fixed threshold τ.
//   Σ-Δ         Manzanera's sigma-delta: M ← M ± 1 toward I, Δ = |I − M|, and a dispersion
//               V ← V ± 1 toward N·Δ (updated every few frames, so slower than M); T = V.
//   Gaussian    running mean and variance, μ ← μ + α (I − μ), σ² ← σ² + α ((I − μ)² − σ²); T = N·σ;
//               α is ten times smaller on pixels found foreground (bi-level update).
//
// Colour frames run one model per channel (B, G, R); a pixel is foreground when the channels'
// mean difference exceeds their mean threshold (grey levels alone confuse an orange object with a
// background of the same brightness).
// Every update is a constant number of operations per pixel and channel: O(W·H·C) per frame, no
// history kept.

#include <vector>

#include <opencv2/core.hpp>

namespace vl::motion {

enum class BgMethod { Mean, Median, SigmaDelta, Gaussian };

struct BgParams {
    BgMethod method = BgMethod::SigmaDelta;
    float alpha = 0.02f;    // learning rate of Mean and Gaussian, in (0, 1]
    float tau = 25.f;       // fixed threshold of Mean and Median, in grey levels
    int amplification = 3;  // N: Σ-Δ makes V follow N·Δ, Gaussian thresholds at N·σ
    int varianceEvery = 2;  // Σ-Δ updates V once every this many frames (V moves slower than M)
    int minVariance = 2;    // Σ-Δ clamps V to [minVariance, 255]
};

// What one frame produced. All images are single-channel with the frame's size; for colour frames,
// difference and threshold are averaged over the channels, and foreground == (difference > threshold).
struct BgStep {
    cv::Mat difference;  // CV_32F, |I_t − M| in grey levels
    cv::Mat threshold;   // CV_32F, the per-pixel threshold T it was compared to
    cv::Mat foreground;  // CV_8U, 255 where difference > threshold
};

class BackgroundModel {
public:
    // background: 8-bit grey or BGR initial estimate M_0 (e.g. the first frame, or medianBackground()).
    // noise: a guess of the sensor noise σ in grey levels, used to start the dispersion of Σ-Δ
    // and Gaussian (they would otherwise flag every pixel until it has grown).
    BackgroundModel(const BgParams& p, const cv::Mat& background, double noise = 2.0);

    // Classify a frame (same type as the initial background) and update the model with it.
    BgStep apply(const cv::Mat& frame);

    cv::Mat background() const;  // the current M, same type as the frames
    const BgParams& params() const { return p_; }
    long frames() const { return t_; }

private:
    BgParams p_;
    // Model state stored as H × (W·C) single-channel planes (channels interleaved like the frame).
    int channels_ = 1;
    cv::Mat m8_, v8_;     // Σ-Δ and Median: M and V as integers (±1 steps)
    cv::Mat mF_, varF_;   // Mean and Gaussian: M and σ² as floats
    long t_ = 0;
};

// Per-pixel (and per-channel) median of `samples` frames spread evenly over the clip. A moving
// object covers any pixel in only a few of them, so the median is the empty scene. O(K·W·H).
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
    double f1() const { return 2.0 * double(tp) + double(fp) + double(fn) > 0
                                   ? 2.0 * double(tp) / (2.0 * double(tp) + double(fp) + double(fn)) : 1.0; }
};
Confusion compareMasks(const cv::Mat& detected, const cv::Mat& truth);

}  // namespace vl::motion
