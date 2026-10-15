#include "app/modules/stereo/stereo_samples.hpp"

#include <cmath>
#include <fstream>

#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>

#include "core/common/synthetic.hpp"
#include "core/stereo/graphcut_stereo.hpp"

namespace vl::app::stereo_samples {
namespace {

Inputs middlebury(const QString& name, const QString& dir) {
    const QString base = dataFile(QStringLiteral("stereo/") + dir);
    const bool ppm = QFileInfo::exists(base + QStringLiteral("/left.ppm"));
    const QString ext = ppm ? QStringLiteral(".ppm") : QStringLiteral(".png");
    Inputs in;
    in.name = name;
    in.images = loadImages({base + "/left" + ext, base + "/right" + ext});
    const cv::Mat raw = cv::imread((base + (ppm ? "/truth.pgm" : "/truth.png")).toStdString(), cv::IMREAD_GRAYSCALE);
    double scale = 1;
    std::ifstream(base.toStdString() + "/scale.txt") >> scale;
    cv::Mat truth;
    raw.convertTo(truth, CV_32F, 1.0 / scale);
    // Middlebury gives no occlusion map: derive it from the true disparities (a left pixel is hidden
    // when a nearer surface lands on its partner), so every stereo topic scores the same pixels.
    cv::Mat visible;
    cv::bitwise_not(vl::stereo::occlusionsFromDisparity(truth), visible);
    visible.setTo(0, truth <= 0);
    in.truth = {truth, visible};
    return in;
}

}  // namespace

std::vector<Sample> all() {
    std::vector<Sample> out;
    out.push_back({QStringLiteral("Synthetic: wall, box and disc"), [] {
                       const auto s = vl::synth::layeredStereo({384, 288}, 32, 7);
                       Inputs in;
                       in.name = QStringLiteral("synthetic");
                       in.images = {s.left, s.right};
                       cv::Mat visible;
                       cv::bitwise_not(s.occluded, visible);
                       in.truth = {s.disparity, visible};
                       return in;
                   }});
    const std::pair<const char*, const char*> sets[] = {
        {"Tsukuba (Middlebury 2001)", "tsukuba"}, {"Cones (Middlebury 2003)", "cones"}, {"Teddy (Middlebury 2003)", "teddy"}};
    for (const auto& [label, dir] : sets) {
        if (dataFile(QStringLiteral("stereo/%1/scale.txt").arg(dir)).isEmpty()) continue;
        const QString l = QString::fromUtf8(label), d = QString::fromUtf8(dir);
        out.push_back({l, [l, d] { return middlebury(l, d); }});
    }
    return out;
}

int maxDisparityHint(const Inputs& in) {
    if (!in.truth.empty() && !in.truth[0].empty()) {
        double hi = 0;
        cv::minMaxLoc(in.truth[0], nullptr, &hi);
        return std::max(8, int(std::ceil(hi / 4.0)) * 4 + 4);
    }
    return std::max(16, in.images[0].cols / 8);
}

}  // namespace vl::app::stereo_samples
