#include "core/common/synthetic.hpp"

#include <cmath>
#include <numbers>
#include <functional>
#include <random>

#include <opencv2/imgproc.hpp>

namespace vl::synth {
namespace {

// Sum of octaves of cubic-upsampled uniform noise (coarse blobs down to 2-pixel grain), normalised to [0, 1].
cv::Mat noiseField(cv::Size size, std::mt19937& rng) {
    cv::Mat acc = cv::Mat::zeros(size, CV_32F);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    float amp = 1.f, total = 0.f;
    for (int cell = 2; cell <= 128; cell *= 2) {
        cv::Mat small(std::max(2, size.height * cell / 256 + 2), std::max(2, size.width * cell / 256 + 2), CV_32F);
        for (auto it = small.begin<float>(); it != small.end<float>(); ++it) *it = u(rng);
        cv::Mat up;
        cv::resize(small, up, size, 0, 0, cv::INTER_CUBIC);
        acc += amp * up;
        total += amp;
        amp *= 0.78f;  // slow decay: enough fine grain to match on, despite sensor noise
    }
    acc /= total;
    cv::normalize(acc, acc, 0, 1, cv::NORM_MINMAX);
    return acc;
}

void strokes(cv::Mat& img, std::mt19937& rng, int count) {
    // Strokes in a much darker or lighter shade of the local tone: strong edges to match on, no
    // random saturated colours.
    std::uniform_int_distribution<int> x(0, img.cols - 1), y(0, img.rows - 1), t(1, 3), dark(0, 1);
    std::uniform_real_distribution<double> lo(0.15, 0.4), hi(1.7, 2.4);
    for (int i = 0; i < count; ++i) {
        const cv::Point a(x(rng), y(rng)), b(x(rng), y(rng));
        const cv::Scalar base = cv::mean(img(cv::Rect(std::min(a.x, b.x), std::min(a.y, b.y),
                                                     std::abs(a.x - b.x) + 1, std::abs(a.y - b.y) + 1)));
        cv::line(img, a, b, base * (dark(rng) ? lo(rng) : hi(rng)), t(rng), cv::LINE_AA);
    }
}

}  // namespace

void addNoise(cv::Mat& img, double sigma, cv::RNG& rng) {
    cv::Mat noise(img.size(), CV_MAKETYPE(CV_32F, img.channels()));
    rng.fill(noise, cv::RNG::NORMAL, cv::Scalar::all(0), cv::Scalar::all(sigma));
    cv::Mat f;
    img.convertTo(f, noise.type());
    cv::Mat(f + noise).convertTo(img, img.type());  // saturating conversion back to 8 bits
}

cv::Mat texture(cv::Size size, std::uint32_t seed) {
    std::mt19937 rng(seed);
    cv::Mat out;
    noiseField(size, rng).convertTo(out, CV_8U, 255.0);
    strokes(out, rng, std::max(4, size.area() / 6000));
    return out;
}

cv::Mat colorTexture(cv::Size size, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<cv::Mat> ch(3);
    for (auto& c : ch) noiseField(size, rng).convertTo(c, CV_8U, 220.0, 20.0);
    cv::Mat out;
    cv::merge(ch, out);
    strokes(out, rng, std::max(4, size.area() / 6000));
    return out;
}

StereoScene layeredStereo(cv::Size size, int maxDisparity, std::uint32_t seed) {
    // Layers back to front. Each has a mask and a disparity function in left coordinates.
    const int W = size.width, H = size.height;
    const double dFar = 0.25 * maxDisparity, dMid = 0.6 * maxDisparity, dNear = 0.95 * maxDisparity;
    auto wallD = [&](double x, double) { return dFar + 0.15 * maxDisparity * x / W; };  // slanted wall
    const cv::Rect box(W / 6, H / 4, W / 3, H / 2);
    const cv::Point2d disc(0.68 * W, 0.55 * H);
    const double radius = 0.17 * std::min(W, H);

    struct Layer {
        cv::Mat tex;
        std::function<bool(double, double)> inside;
        std::function<double(double, double)> disp;
    };
    std::vector<Layer> layers = {
        {colorTexture(size + cv::Size(maxDisparity + 2, 0), seed), [](double, double) { return true; }, wallD},
        {colorTexture(size + cv::Size(maxDisparity + 2, 0), seed + 1),
         [&](double x, double y) { return box.contains(cv::Point(int(x), int(y))); },
         [&](double, double) { return dMid; }},
        {colorTexture(size + cv::Size(maxDisparity + 2, 0), seed + 2),
         [&](double x, double y) { return std::hypot(x - disc.x, y - disc.y) < radius; },
         [&](double, double) { return dNear; }},
    };

    StereoScene s;
    s.maxDisparity = maxDisparity;
    s.left.create(size, CV_8UC3);
    s.right.create(size, CV_8UC3);
    s.disparity.create(size, CV_32F);
    // Left view: the frontmost layer covering each pixel.
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int k = int(layers.size()) - 1; k >= 0; --k)
                if (layers[k].inside(x, y)) {
                    s.left.at<cv::Vec3b>(y, x) = layers[k].tex.at<cv::Vec3b>(y, x);
                    s.disparity.at<float>(y, x) = float(layers[k].disp(x, y));
                    break;
                }
    // Right view: right (xr, y) sees left point x = xr + d. Take the frontmost layer that lands there.
    for (int y = 0; y < H; ++y)
        for (int xr = 0; xr < W; ++xr)
            for (int k = int(layers.size()) - 1; k >= 0; --k) {
                double d = layers[k].disp(xr, y);
                d = layers[k].disp(xr + d, y);  // one fixed-point step for the slanted wall
                const double xl = xr + d;
                if (layers[k].inside(xl, y)) {
                    const int xi = std::clamp(int(std::lround(xl)), 0, layers[k].tex.cols - 1);
                    s.right.at<cv::Vec3b>(y, xr) = layers[k].tex.at<cv::Vec3b>(y, xi);
                    break;
                }
            }
    // A left pixel is occluded when the right view shows a nearer surface at x - d.
    s.occluded = cv::Mat::zeros(size, CV_8U);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float d = s.disparity.at<float>(y, x);
            const int xr = int(std::lround(float(x) - d));
            if (xr < 0) { s.occluded.at<uchar>(y, x) = 255; continue; }
            for (int k = int(layers.size()) - 1; k >= 0; --k) {
                const double dk = layers[k].disp(xr, y);
                if (layers[k].inside(xr + dk, y)) {
                    if (dk > d + 0.5) s.occluded.at<uchar>(y, x) = 255;
                    break;
                }
            }
        }
    // Mild sensor noise, different in each view.
    cv::RNG rng(seed + 99);
    addNoise(s.left, 2.0, rng);
    addNoise(s.right, 2.0, rng);
    return s;
}

cv::Mat warpBackward(const cv::Mat& img, const cv::Mat& flow) {
    CV_Assert(flow.type() == CV_32FC2 && flow.size() == img.size());
    cv::Mat map(flow.size(), CV_32FC2);
    for (int y = 0; y < flow.rows; ++y)
        for (int x = 0; x < flow.cols; ++x) {
            const auto f = flow.at<cv::Vec2f>(y, x);
            map.at<cv::Vec2f>(y, x) = {float(x) + f[0], float(y) + f[1]};
        }
    cv::Mat out;
    cv::remap(img, out, map, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}

FlowPair rigidFlow(cv::Size size, double angleDeg, double zoom, cv::Point2d shift, std::uint32_t seed) {
    FlowPair p;
    p.frame0 = colorTexture(size, seed);
    const cv::Point2d c(size.width / 2.0, size.height / 2.0);
    const double a = angleDeg * std::numbers::pi / 180.0, ca = std::cos(a) * zoom, sa = std::sin(a) * zoom;
    // Forward motion of a frame0 point: p1 = c + R (p0 - c) + shift.
    p.flow.create(size, CV_32FC2);
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x) {
            const double dx = x - c.x, dy = y - c.y;
            const double x1 = c.x + ca * dx - sa * dy + shift.x, y1 = c.y + sa * dx + ca * dy + shift.y;
            p.flow.at<cv::Vec2f>(y, x) = {float(x1 - x), float(y1 - y)};
        }
    // frame1 (q) = frame0 (inverse motion of q): build the inverse field and warp backward.
    const double det = ca * ca + sa * sa;
    cv::Mat inv(size, CV_32FC2);
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x) {
            const double qx = x - c.x - shift.x, qy = y - c.y - shift.y;
            const double x0 = c.x + (ca * qx + sa * qy) / det, y0 = c.y + (-sa * qx + ca * qy) / det;
            inv.at<cv::Vec2f>(y, x) = {float(x0 - x), float(y0 - y)};
        }
    p.frame1 = warpBackward(p.frame0, inv);
    return p;
}

MovingScene movingObject(cv::Size size, int nFrames, std::uint32_t seed) {
    MovingScene m;
    std::mt19937 rng(seed);
    cv::Mat bg = colorTexture(size, seed);
    bg = bg * 0.6 + cv::Scalar(40, 40, 40);  // a calmer background, so the object stands out
    const int r = std::max(6, std::min(size.width, size.height) / 10);
    cv::Mat obj = colorTexture({2 * r + 1, 2 * r + 1}, seed + 7);
    cv::Mat objMask = cv::Mat::zeros(obj.size(), CV_8U);
    cv::circle(objMask, {r, r}, r, 255, cv::FILLED, cv::LINE_AA);
    cv::Mat tint(obj.size(), CV_8UC3, cv::Scalar(30, 90, 235));
    cv::addWeighted(obj, 0.35, tint, 0.65, 0, obj);  // warm-coloured object: easy for hue tracking

    cv::RNG noiseRng(seed + 99);
    for (int t = 0; t < nFrames; ++t) {
        const double s = double(t) / std::max(1, nFrames - 1) * 2 * std::numbers::pi;
        const cv::Point centre(int(size.width * (0.5 + 0.33 * std::sin(s))),
                               int(size.height * (0.5 + 0.28 * std::sin(2 * s + 0.6))));
        cv::Mat frame = bg * (1.0 + 0.04 * std::sin(0.7 * s));  // slow lighting drift
        cv::Mat mask = cv::Mat::zeros(size, CV_8U);
        const cv::Rect box(centre.x - r, centre.y - r, 2 * r + 1, 2 * r + 1);
        const cv::Rect clip = box & cv::Rect({0, 0}, size);
        if (clip.area() > 0) {
            const cv::Rect src(clip.tl() - box.tl(), clip.size());
            obj(src).copyTo(frame(clip), objMask(src));
            objMask(src).copyTo(mask(clip));
        }
        addNoise(frame, 3.0, noiseRng);
        m.frames.push_back(frame);
        m.masks.push_back(mask);
        m.boxes.push_back(clip);
    }
    return m;
}

}  // namespace vl::synth
