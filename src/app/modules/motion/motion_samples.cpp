#include "app/modules/motion/motion_samples.hpp"

#include <algorithm>

#include <QFileInfo>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "core/common/image.hpp"
#include "core/common/synthetic.hpp"
#include "core/motion/sequences.hpp"

namespace vl::app::motion_samples {
namespace {

constexpr int kMaxSide = 320;    // keeps a 300-frame clip interactive and its views in memory
constexpr int kMaxFrames = 300;
const cv::Size kSynthSize{320, 240};

Inputs video(const QString& name, const QString& file) {
    const QString path = dataFile(QStringLiteral("video/") + file);
    cv::VideoCapture cap(path.toStdString());
    if (!cap.isOpened()) throw std::runtime_error("cannot open " + path.toStdString());
    Inputs in;
    in.name = name;
    in.fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 25.0;
    cv::Mat f;
    while (int(in.images.size()) < kMaxFrames && cap.read(f)) in.images.push_back(vl::limitSize(f, kMaxSide).clone());
    if (in.images.empty()) throw std::runtime_error("no frames in " + path.toStdString());
    return in;
}

Inputs fromScene(const QString& name, vl::synth::MovingScene s, bool boxesAsTruth) {
    Inputs in;
    in.name = name;
    in.images = std::move(s.frames);
    if (boxesAsTruth) {
        for (const cv::Rect& b : s.boxes) {
            cv::Mat m = cv::Mat::zeros(in.images.front().size(), CV_8U);
            m(b & cv::Rect({0, 0}, m.size())).setTo(255);
            in.truth.push_back(m);
        }
    } else {
        in.truth = std::move(s.masks);
    }
    return in;
}

Sample synthMoving(bool boxes) {
    return {QStringLiteral("Synthetic: orange disc on texture"), [boxes] {
                auto s = vl::synth::movingObject(kSynthSize, boxes ? 120 : 150, 11);
                vl::motion::addSensorNoise(s.frames, 3.0, 11);
                return fromScene(QStringLiteral("synthetic"), std::move(s), boxes);
            }};
}

Sample synthOccluded(bool boxes) {
    return {QStringLiteral("Synthetic: disc behind a pillar"), [boxes] {
                return fromScene(QStringLiteral("synthetic-occluded"), vl::motion::occludedCrossing(kSynthSize, 90, 5), boxes);
            }};
}

struct Clip {
    const char* label;
    const char* file;
    QRectF box;  // target on the first frame, in pixels of the reduced (≤ 320 px) frames
};

// The tracking practical's clips. Boxes picked by eye on the first frame.
const Clip kClips[] = {
    {"Mug (static camera)", "Antoine_Mug.mp4", {88, 90, 56, 72}},
    {"Ball (VOT)", "VOT-Ball.mp4", {198, 112, 48, 48}},
    {"Car (VOT)", "VOT-Car.mp4", {122, 83, 58, 55}},
    {"Woman (VOT)", "VOT-Woman.mp4", {183, 106, 24, 94}},
    {"Sunshade (VOT)", "VOT-Sunshade.mp4", {68, 82, 28, 34}},
    {"Basketball (VOT)", "VOT-Basket.mp4", {150, 96, 22, 60}},
};

void addClips(std::vector<Sample>& out, std::initializer_list<const char*> files) {
    for (const char* f : files)
        for (const Clip& c : kClips) {
            if (QString::fromUtf8(c.file) != QString::fromUtf8(f)) continue;
            if (dataFile(QStringLiteral("video/") + f).isEmpty()) continue;
            const QString label = QString::fromUtf8(c.label), file = QString::fromUtf8(c.file);
            out.push_back({label, [label, file] { return video(label, file); }});
        }
}

}  // namespace

std::vector<Sample> background() {
    std::vector<Sample> out{synthMoving(false), synthOccluded(false)};
    // Only the mug clip has a camera that stays put; the VOT clips pan or are handheld, which
    // breaks the static-background assumption (Sunshade is kept to show it).
    addClips(out, {"Antoine_Mug.mp4", "VOT-Sunshade.mp4"});
    return out;
}

std::vector<Sample> tracking() {
    std::vector<Sample> out{synthMoving(true), synthOccluded(true)};
    addClips(out, {"VOT-Ball.mp4", "Antoine_Mug.mp4", "VOT-Car.mp4", "VOT-Woman.mp4", "VOT-Sunshade.mp4", "VOT-Basket.mp4"});
    return out;
}

cv::Rect2d truthBox(const Inputs& in, size_t t) {
    if (t >= in.truth.size() || in.truth[t].empty()) return {};
    // Bounding box of the non-zero pixels (cv::boundingRect moved modules in OpenCV 5).
    std::vector<cv::Point> pts;
    cv::findNonZero(in.truth[t], pts);
    if (pts.empty()) return {};
    int x0 = pts[0].x, x1 = x0, y0 = pts[0].y, y1 = y0;
    for (const cv::Point& q : pts) {
        x0 = std::min(x0, q.x);
        x1 = std::max(x1, q.x);
        y0 = std::min(y0, q.y);
        y1 = std::max(y1, q.y);
    }
    return cv::Rect2d(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
}

StartBox defaultBox(const Inputs& in) {
    if (const cv::Rect2d b = truthBox(in, 0); b.area() > 0) return {QRectF(b.x, b.y, b.width, b.height), true};
    for (const Clip& c : kClips)
        if (in.name == QString::fromUtf8(c.label)) return {c.box, true};
    const cv::Size s = in.images.front().size();
    return {QRectF(s.width * 0.4, s.height * 0.35, s.width * 0.2, s.height * 0.3), false};
}

}  // namespace vl::app::motion_samples
