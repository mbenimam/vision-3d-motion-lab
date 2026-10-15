#include "core/motion/sequences.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace vl::motion {

synth::MovingScene occludedCrossing(cv::Size size, int nFrames, std::uint32_t seed) {
    CV_Assert(nFrames >= 2);
    synth::MovingScene m;
    // Calm, desaturated background so the hue model sees the object only.
    cv::Mat bg = synth::colorTexture(size, seed);
    cv::Mat grey;
    cv::cvtColor(bg, grey, cv::COLOR_BGR2GRAY);
    cv::cvtColor(grey, grey, cv::COLOR_GRAY2BGR);
    cv::addWeighted(bg, 0.25, grey, 0.75, 0, bg);
    bg = bg * 0.6 + cv::Scalar(45, 45, 45);

    const int r = std::max(6, std::min(size.width, size.height) / 10);
    cv::Mat obj = synth::colorTexture({2 * r + 1, 2 * r + 1}, seed + 7);
    cv::addWeighted(obj, 0.35, cv::Mat(obj.size(), CV_8UC3, cv::Scalar(30, 90, 235)), 0.65, 0, obj);
    cv::Mat objMask = cv::Mat::zeros(obj.size(), CV_8U);
    cv::circle(objMask, {r, r}, r, 255, cv::FILLED, cv::LINE_AA);

    // Pillar: a vertical band of neutral grey with a little texture, a bit wider than the object.
    const int pw = 2 * r + r / 2;
    const cv::Rect pillar(size.width / 2 - pw / 2, 0, pw, size.height);
    cv::Mat pillarTex = synth::texture(pillar.size(), seed + 13);
    cv::Mat pillarImg;
    cv::cvtColor(pillarTex * 0.35 + 90, pillarImg, cv::COLOR_GRAY2BGR);

    const cv::Point2d from(0.1 * size.width, 0.38 * size.height), to(0.9 * size.width, 0.62 * size.height);
    for (int t = 0; t < nFrames; ++t) {
        const double u = double(t) / double(nFrames - 1);
        const cv::Point c(int(std::lround(from.x + u * (to.x - from.x))), int(std::lround(from.y + u * (to.y - from.y))));
        cv::Mat frame = bg.clone();
        cv::Mat mask = cv::Mat::zeros(size, CV_8U);
        const cv::Rect box(c.x - r, c.y - r, 2 * r + 1, 2 * r + 1);
        const cv::Rect clip = box & cv::Rect({0, 0}, size);
        if (clip.area() > 0) {
            const cv::Rect src(clip.tl() - box.tl(), clip.size());
            obj(src).copyTo(frame(clip), objMask(src));
            objMask(src).copyTo(mask(clip));
        }
        pillarImg.copyTo(frame(pillar));  // in front of the object
        mask(pillar).setTo(0);
        m.frames.push_back(frame);
        m.masks.push_back(mask);
        m.boxes.push_back(box);
    }
    addSensorNoise(m.frames, 3.0, seed);
    return m;
}

void addSensorNoise(std::vector<cv::Mat>& frames, double sigma, std::uint32_t seed) {
    cv::RNG rng(seed);
    cv::Mat n, f16;
    for (cv::Mat& f : frames) {
        CV_Assert(f.depth() == CV_8U);
        n.create(f.size(), CV_16SC(f.channels()));
        rng.fill(n, cv::RNG::NORMAL, cv::Scalar::all(0), cv::Scalar::all(sigma));
        f.convertTo(f16, n.type());
        cv::Mat(f16 + n).convertTo(f, f.type());
    }
}

}  // namespace vl::motion
