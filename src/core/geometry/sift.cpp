// SIFT, the scale-invariant feature transform (Lowe 2004). Course "3D Vision and Motion", part 3.
//
// The method in seven steps, one function each:
//   1. buildScaleSpace    L(x, y, σ) = G_σ * I for σ = σ₀ k^i, k = 2^(1/s): octaves of s + 3 blurred
//                         layers, each octave half the size of the previous one; and the differences
//                         D(x, y, σ) = L(x, y, kσ) − L(x, y, σ) ≈ (k − 1) σ² ∇²G * I, a blob detector
//   2. findExtrema        samples of D larger (or smaller) than their 26 neighbours in x, y and scale
//   3. refineExtremum     fit D(x₀ + x̂) ≈ D + ∇Dᵀ x̂ + ½ x̂ᵀ H x̂ and move to its extremum x̂ = −H⁻¹ ∇D
//                         (sub-pixel, sub-scale); drop low contrast |D(x̂)| and edges (isOnEdge)
//   4. computeGradients   magnitude m and orientation θ of every layer keypoints live on, once
//   5. assignOrientation  36-bin histogram of θ around the keypoint weighted by m and a Gaussian; every
//                         peak ≥ 80 % of the highest gives a keypoint (angle refined by a parabola)
//   6. keepStrongest      the maxFeatures keypoints of highest contrast
//   7. describe           4 × 4 cells × 8 orientations of the gradients around the keypoint, in its own
//                         rotated and scaled frame, trilinearly interpolated; normalised, clipped at
//                         0.2 and renormalised (robust to lighting)
//
// Cost: the scale space is O(W·H·Σ blur radii), the smaller octaves adding only 1/4 + 1/16 + ... ≈ 1/3;
// each descriptor reads a disc of radius 3σ · √2 · 2.5 pixels of its octave. Together a few hundred
// milliseconds for a 640×480 image.

#include "core/geometry/sift.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <optional>

#include "core/common/filters.hpp"
#include "core/common/image.hpp"
#include "core/common/task.hpp"

namespace vl::geometry {
namespace {

constexpr double kTwoPi = 2 * std::numbers::pi;
constexpr int kBorder = 5;             // extrema this close to the octave's border are not searched
constexpr int kMinOctaveSide = 16;     // stop halving below this
constexpr int kRefineSteps = 5;        // moves allowed while refining before giving up
constexpr int kOrientationBins = 36;   // 10° each
constexpr double kPeakRatio = 0.8;     // secondary orientations within 80 % of the main one
constexpr int kCells = 4;              // the descriptor is kCells × kCells cells ...
constexpr int kAngleBins = 8;          // ... of kAngleBins orientations: 4 · 4 · 8 = 128 numbers
constexpr int kDescriptorSize = kCells * kCells * kAngleBins;
constexpr double kClip = 0.2;          // no single gradient direction may dominate the descriptor

// One octave of the scale space, in its own pixels.
struct Octave {
    std::vector<cv::Mat> L;  // s + 3 Gaussian layers: L[i] has blur σ₀ k^i
    std::vector<cv::Mat> D;  // s + 2 differences: D[i] = L[i + 1] − L[i]
    std::vector<cv::Mat> magnitude, orientation;  // gradients of L[i], for i = 1..s (step 4)
    double pixelSize = 1;    // input pixels per octave pixel: 2^o (2^(o−1) when upsampled)
};

// A candidate extremum: octave pixel (x, y) of difference layer `layer`.
struct Sample {
    int x = 0, y = 0, layer = 0;
};

// The image doubled in size: out(x, y) = I(x / 2, y / 2), bilinear.
cv::Mat upsampleTwice(const cv::Mat& I) {
    cv::Mat out(2 * I.rows, 2 * I.cols, CV_32F);
    for (int y = 0; y < out.rows; ++y)
        for (int x = 0; x < out.cols; ++x) out.at<float>(y, x) = bilinear(I, 0.5f * float(x), 0.5f * float(y));
    return out;
}

// Every other row and column. Used on a layer blurred enough (2σ₀) that nothing aliases.
cv::Mat halve(const cv::Mat& f) {
    cv::Mat out(f.rows / 2, f.cols / 2, CV_32F);
    for (int y = 0; y < out.rows; ++y)
        for (int x = 0; x < out.cols; ++x) out.at<float>(y, x) = f.at<float>(2 * y, 2 * x);
    return out;
}

// The input in [0, 1], blurred to σ₀. A camera image already has a blur of about 0.5 pixel (1 after
// doubling) and blurs compose as σ² = σa² + σb², so only √(σ₀² − σ_present²) is added.
cv::Mat baseImage(const cv::Mat& image, const SiftParams& p) {
    cv::Mat I = toGrayF(image) / 255.0;
    double present = 0.5;
    if (p.upsample) {
        I = upsampleTwice(I);
        present = 1.0;
    }
    return gaussianBlur(I, std::sqrt(p.sigma0 * p.sigma0 - present * present));
}

// Step 1. Each layer is the previous one blurred by the missing amount √(σᵢ² − σᵢ₋₁²): small kernels.
std::vector<Octave> buildScaleSpace(const cv::Mat& image, const SiftParams& p) {
    const int s = p.intervals;
    const double k = std::pow(2.0, 1.0 / s);
    cv::Mat base = baseImage(image, p);
    double pixelSize = p.upsample ? 0.5 : 1.0;
    std::vector<Octave> octaves;
    while (std::min(base.rows, base.cols) >= kMinOctaveSide) {
        Octave o;
        o.pixelSize = pixelSize;
        o.L.push_back(base);
        for (int i = 1; i < s + 3; ++i) {
            const double before = p.sigma0 * std::pow(k, i - 1), after = before * k;
            o.L.push_back(gaussianBlur(o.L.back(), std::sqrt(after * after - before * before)));
        }
        for (int i = 0; i + 1 < s + 3; ++i) o.D.push_back(o.L[size_t(i + 1)] - o.L[size_t(i)]);
        // L[s] has blur 2σ₀; every other pixel of it has blur σ₀ in the next octave's pixels.
        base = halve(o.L[size_t(s)]);
        pixelSize *= 2;
        octaves.push_back(std::move(o));
    }
    return octaves;
}

// Is D[layer](x, y) strictly above, or strictly below, all 26 neighbours (8 in its layer, 9 in each
// of the layers above and below)?
bool isExtremum(const Octave& o, int layer, int y, int x) {
    const float v = o.D[size_t(layer)].at<float>(y, x);
    bool isMax = true, isMin = true;
    for (int l = layer - 1; l <= layer + 1; ++l)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (l == layer && dy == 0 && dx == 0) continue;
                const float w = o.D[size_t(l)].at<float>(y + dy, x + dx);
                isMax = isMax && v > w;
                isMin = isMin && v < w;
                if (!isMax && !isMin) return false;
            }
    return true;
}

// Step 2, on the layers 1..s (each needs a layer above and below). A sample below half the final
// contrast threshold cannot pass it after refinement, so it is skipped before the 26 comparisons.
std::vector<Sample> findExtrema(const Octave& o, const SiftParams& p) {
    const double prefilter = 0.5 * p.contrastThreshold / p.intervals;
    std::vector<Sample> out;
    for (int layer = 1; layer <= p.intervals; ++layer) {
        const cv::Mat& D = o.D[size_t(layer)];
        for (int y = kBorder; y < D.rows - kBorder; ++y)
            for (int x = kBorder; x < D.cols - kBorder; ++x)
                if (std::abs(D.at<float>(y, x)) > prefilter && isExtremum(o, layer, y, x)) out.push_back({x, y, layer});
    }
    return out;
}

// D around a sample as a quadratic: value, gradient ∇D and Hessian H in (x, y, layer), by central
// finite differences.
struct Quadratic {
    double value = 0;
    cv::Vec3d gradient;
    cv::Matx33d hessian;
};

Quadratic localQuadratic(const Octave& o, const Sample& c) {
    auto D = [&](int dl, int dy, int dx) { return double(o.D[size_t(c.layer + dl)].at<float>(c.y + dy, c.x + dx)); };
    Quadratic q;
    q.value = D(0, 0, 0);
    q.gradient = {(D(0, 0, 1) - D(0, 0, -1)) / 2, (D(0, 1, 0) - D(0, -1, 0)) / 2, (D(1, 0, 0) - D(-1, 0, 0)) / 2};
    const double Dxx = D(0, 0, 1) + D(0, 0, -1) - 2 * q.value;
    const double Dyy = D(0, 1, 0) + D(0, -1, 0) - 2 * q.value;
    const double Dss = D(1, 0, 0) + D(-1, 0, 0) - 2 * q.value;
    const double Dxy = (D(0, 1, 1) - D(0, 1, -1) - D(0, -1, 1) + D(0, -1, -1)) / 4;
    const double Dxs = (D(1, 0, 1) - D(1, 0, -1) - D(-1, 0, 1) + D(-1, 0, -1)) / 4;
    const double Dys = (D(1, 1, 0) - D(1, -1, 0) - D(-1, 1, 0) + D(-1, -1, 0)) / 4;
    q.hessian = cv::Matx33d(Dxx, Dxy, Dxs, Dxy, Dyy, Dys, Dxs, Dys, Dss);
    return q;
}

// Edge test on the 2×2 spatial Hessian [Dxx Dxy; Dxy Dyy], whose eigenvalues α ≥ β are the principal
// curvatures. Along an edge α ≫ β, and with α = r β, Tr² / Det = (α + β)² / (α β) = (r + 1)² / r, which
// grows with r: comparing it to the bound needs no eigenvalue. Det ≤ 0: curvatures of opposite signs.
bool isOnEdge(const Quadratic& q, double r) {
    const double Dxx = q.hessian(0, 0), Dyy = q.hessian(1, 1), Dxy = q.hessian(0, 1);
    const double trace = Dxx + Dyy, det = Dxx * Dyy - Dxy * Dxy;
    return det <= 0 || trace * trace * r >= (r + 1) * (r + 1) * det;
}

// Step 3. When an offset exceeds half a sample the extremum is nearer the neighbour: move one sample
// that way and fit again. Points that keep moving, or leave the octave, are unstable and dropped.
std::optional<Keypoint> refineExtremum(const Octave& o, Sample c, int octaveIndex, const SiftParams& p) {
    const cv::Size size = o.D.front().size();
    Quadratic q;
    cv::Vec3d xhat;  // offset (x, y, layer) from the sample to the fitted extremum
    for (int step = 0;; ++step) {
        if (step == kRefineSteps) return std::nullopt;
        q = localQuadratic(o, c);
        // x̂ = −H⁻¹ ∇D: where the gradient of the quadratic vanishes.
        if (std::abs(cv::determinant(q.hessian)) < 1e-12) return std::nullopt;
        xhat = q.hessian.solve(-q.gradient, cv::DECOMP_LU);
        if (std::abs(xhat[0]) <= 0.5 && std::abs(xhat[1]) <= 0.5 && std::abs(xhat[2]) <= 0.5) break;
        if (std::abs(xhat[0]) > 0.5) c.x += xhat[0] > 0 ? 1 : -1;
        if (std::abs(xhat[1]) > 0.5) c.y += xhat[1] > 0 ? 1 : -1;
        if (std::abs(xhat[2]) > 0.5) c.layer += xhat[2] > 0 ? 1 : -1;
        if (c.layer < 1 || c.layer > p.intervals || c.x < kBorder || c.y < kBorder || c.x >= size.width - kBorder ||
            c.y >= size.height - kBorder)
            return std::nullopt;
    }
    // D(x̂) = D + ½ ∇Dᵀ x̂: the value at the extremum of the quadratic. Low: noise, not structure.
    const double contrast = std::abs(q.value + 0.5 * q.gradient.dot(xhat));
    if (contrast < p.contrastThreshold / p.intervals) return std::nullopt;
    if (isOnEdge(q, p.edgeRatio)) return std::nullopt;

    Keypoint kp;
    kp.octave = octaveIndex;
    kp.layer = c.layer + xhat[2];
    kp.pt = cv::Point2d((c.x + xhat[0]) * o.pixelSize, (c.y + xhat[1]) * o.pixelSize);
    kp.sigma = p.sigma0 * std::pow(2.0, kp.layer / p.intervals) * o.pixelSize;
    kp.contrast = contrast;
    return kp;
}

// Step 4: m = √(Lx² + Ly²) and θ = atan2(Ly, Lx) ∈ [0, 2π), central differences. Orientation and
// descriptor read them many times over (overlapping windows): computed once per layer 1..s.
void computeGradients(Octave& o, int s) {
    o.magnitude.assign(o.L.size(), cv::Mat());
    o.orientation.assign(o.L.size(), cv::Mat());
    for (int i = 1; i <= s; ++i) {
        const cv::Mat& L = o.L[size_t(i)];
        cv::Mat m(L.size(), CV_32F), theta(L.size(), CV_32F);
        for (int y = 0; y < L.rows; ++y)
            for (int x = 0; x < L.cols; ++x) {
                const double Lx = at(L, y, x + 1) - at(L, y, x - 1), Ly = at(L, y + 1, x) - at(L, y - 1, x);
                double angle = std::atan2(Ly, Lx);
                if (angle < 0) angle += kTwoPi;
                m.at<float>(y, x) = float(std::sqrt(Lx * Lx + Ly * Ly));
                theta.at<float>(y, x) = float(angle);
            }
        o.magnitude[size_t(i)] = m;
        o.orientation[size_t(i)] = theta;
    }
}

// Where a keypoint sits in its octave: nearest pixel and layer, and its σ in octave pixels.
struct Place {
    int x = 0, y = 0, layer = 0;
    double sigma = 0;
};

Place placeInOctave(const Octave& o, const Keypoint& kp, int s) {
    return {int(std::lround(kp.pt.x / o.pixelSize)), int(std::lround(kp.pt.y / o.pixelSize)),
            std::clamp(int(std::lround(kp.layer)), 1, s), kp.sigma / o.pixelSize};
}

// Circular smoothing with [1 4 6 4 1] / 16, so a peak split between two bins still shows as one.
void smoothCircular(std::vector<double>& h) {
    const int n = int(h.size());
    const std::vector<double> in = h;
    for (int i = 0; i < n; ++i)
        h[size_t(i)] = (in[size_t((i - 2 + n) % n)] + in[size_t((i + 2) % n)] +
                        4 * (in[size_t((i - 1 + n) % n)] + in[size_t((i + 1) % n)]) + 6 * in[size_t(i)]) / 16;
}

// Step 5. Gradients in a disc of radius 3 · 1.5σ, weighted by m and by a Gaussian of σ_w = 1.5σ.
std::vector<double> assignOrientation(const Octave& o, const Keypoint& kp, int s) {
    const Place c = placeInOctave(o, kp, s);
    const cv::Mat& m = o.magnitude[size_t(c.layer)];
    const cv::Mat& theta = o.orientation[size_t(c.layer)];
    const double sigmaW = 1.5 * c.sigma;
    const int radius = int(std::lround(3 * sigmaW));
    std::vector<double> h(kOrientationBins, 0.0);
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            const int y = c.y + dy, x = c.x + dx;
            if (y < 0 || x < 0 || y >= m.rows || x >= m.cols) continue;
            const double w = std::exp(-(dx * dx + dy * dy) / (2 * sigmaW * sigmaW));
            const int bin = std::min(kOrientationBins - 1, int(theta.at<float>(y, x) * kOrientationBins / kTwoPi));
            h[size_t(bin)] += w * m.at<float>(y, x);
        }
    smoothCircular(h);

    const double highest = *std::max_element(h.begin(), h.end());
    std::vector<double> angles;
    for (int i = 0; i < kOrientationBins; ++i) {
        const double l = h[size_t((i - 1 + kOrientationBins) % kOrientationBins)], c0 = h[size_t(i)];
        const double r = h[size_t((i + 1) % kOrientationBins)];
        if (c0 <= l || c0 <= r || c0 < kPeakRatio * highest) continue;
        // The parabola through the three bins peaks at i + ½ (l − r) / (l − 2c + r); bin i is centred on i + ½.
        const double offset = 0.5 * (l - r) / (l - 2 * c0 + r);
        double angle = (i + 0.5 + offset) * kTwoPi / kOrientationBins;
        angle = std::fmod(angle + kTwoPi, kTwoPi);
        angles.push_back(angle);
    }
    return angles;
}

// Step 6. The strongest extrema are the most repeatable ones.
void keepStrongest(std::vector<Keypoint>& keypoints, int maxFeatures) {
    if (maxFeatures <= 0 || int(keypoints.size()) <= maxFeatures) return;
    std::stable_sort(keypoints.begin(), keypoints.end(),
                     [](const Keypoint& a, const Keypoint& b) { return a.contrast > b.contrast; });
    keypoints.resize(size_t(maxFeatures));
}

// Adds `value` to the 4 × 4 × 8 histogram at the real position (row r, column c, orientation b),
// shared among the 8 surrounding bins in proportion to closeness (trilinear interpolation): a
// gradient moving slightly changes the descriptor slightly, never by a whole bin.
void addTrilinear(std::vector<float>& h, double r, double c, double b, double value) {
    const int r0 = int(std::floor(r)), c0 = int(std::floor(c)), b0 = int(std::floor(b));
    const double fr = r - r0, fc = c - c0, fb = b - b0;
    for (int i = 0; i <= 1; ++i)
        for (int j = 0; j <= 1; ++j)
            for (int k = 0; k <= 1; ++k) {
                const int row = r0 + i, col = c0 + j, bin = (b0 + k) % kAngleBins;  // orientation wraps around
                if (row < 0 || row >= kCells || col < 0 || col >= kCells) continue;
                const double w = (i ? fr : 1 - fr) * (j ? fc : 1 - fc) * (k ? fb : 1 - fb);
                h[size_t((row * kCells + col) * kAngleBins + bin)] += float(w * value);
            }
}

// v / ‖v‖ (left as is when zero).
void normalise(std::vector<float>& v) {
    double norm = 0;
    for (float x : v) norm += double(x) * x;
    norm = std::sqrt(norm);
    if (norm > 0)
        for (float& x : v) x = float(x / norm);
}

// Step 7. Each cell is 3σ wide; a sample at offset (dx, dy) from the keypoint is rotated by −angle and
// divided by the cell width, giving (u, v) in cells, then weighted by a Gaussian of half the
// descriptor's width. Its orientation is taken relative to the keypoint's: the descriptor turns with it.
std::vector<float> describe(const Octave& o, const Keypoint& kp, int s) {
    const Place c = placeInOctave(o, kp, s);
    const cv::Mat& m = o.magnitude[size_t(c.layer)];
    const cv::Mat& theta = o.orientation[size_t(c.layer)];
    const double cellWidth = 3 * c.sigma;
    // The farthest sample that can still touch a cell: half the diagonal of (kCells + 1) cells.
    const int radius = int(std::lround(cellWidth * std::sqrt(2.0) * (kCells + 1) / 2));
    const double cosA = std::cos(kp.angle), sinA = std::sin(kp.angle);
    const double halfWidth = kCells / 2.0;

    std::vector<float> h(kDescriptorSize, 0.f);
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            const double u = (cosA * dx + sinA * dy) / cellWidth, v = (-sinA * dx + cosA * dy) / cellWidth;
            // Cell coordinates with cell centres at 0, 1, 2, 3.
            const double col = u + halfWidth - 0.5, row = v + halfWidth - 0.5;
            if (row <= -1 || row >= kCells || col <= -1 || col >= kCells) continue;
            const int y = c.y + dy, x = c.x + dx;
            if (y < 0 || x < 0 || y >= m.rows || x >= m.cols) continue;
            const double w = std::exp(-(u * u + v * v) / (2 * halfWidth * halfWidth));
            double relative = theta.at<float>(y, x) - kp.angle;
            if (relative < 0) relative += kTwoPi;
            addTrilinear(h, row, col, relative * kAngleBins / kTwoPi, w * m.at<float>(y, x));
        }
    // Unit length cancels a contrast change; clipping limits what a few strong (lighting) edges can
    // weigh; renormalising restores unit length.
    normalise(h);
    for (float& x : h) x = std::min(x, float(kClip));
    normalise(h);
    return h;
}

}  // namespace

Features detectSift(const cv::Mat& image, const SiftParams& p, std::stop_token stop) {
    CV_Assert(p.intervals >= 1 && p.sigma0 > 0.5);
    // 1. The scale space: Gaussian octaves and their differences.
    std::vector<Octave> octaves = buildScaleSpace(image, p);
    checkStop(stop);

    std::vector<Keypoint> keypoints;
    for (int o = 0; o < int(octaves.size()); ++o) {
        Octave& octave = octaves[size_t(o)];
        // 4. Gradients first: orientation (5) needs them right after detection.
        computeGradients(octave, p.intervals);
        // 2–3. Extrema of D, refined; low-contrast and edge responses dropped.
        for (const Sample& c : findExtrema(octave, p)) {
            const auto kp = refineExtremum(octave, c, o, p);
            if (!kp) continue;
            // 5. One keypoint per dominant orientation.
            for (double angle : assignOrientation(octave, *kp, p.intervals)) {
                keypoints.push_back(*kp);
                keypoints.back().angle = angle;
            }
        }
        checkStop(stop);
    }
    // 6. The strongest ones.
    keepStrongest(keypoints, p.maxFeatures);

    // 7. One descriptor per keypoint, in its own octave.
    Features f;
    f.descriptors.create(int(keypoints.size()), kDescriptorSize, CV_32F);
    for (int i = 0; i < int(keypoints.size()); ++i) {
        if (i % 256 == 0) checkStop(stop);
        const Keypoint& kp = keypoints[size_t(i)];
        const std::vector<float> d = describe(octaves[size_t(kp.octave)], kp, p.intervals);
        for (int k = 0; k < kDescriptorSize; ++k) f.descriptors.at<float>(i, k) = d[size_t(k)];
    }
    f.keypoints = std::move(keypoints);
    return f;
}

}  // namespace vl::geometry
