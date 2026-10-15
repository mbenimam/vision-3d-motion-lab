// Topic: tracking by mean shift on the back-projection of a hue histogram, optionally with a
// constant-velocity Kalman prediction. Baseline: cv::meanShift or cv::CamShift climbing the same
// back-projection from the same start box.

#include <cmath>
#include <optional>

#include <opencv2/imgproc.hpp>

#include "app/module.hpp"
#include "app/modules/motion/motion_samples.hpp"
#include "app/modules/motion/tracking_common.hpp"
#include "core/common/image.hpp"
#include "core/motion/mean_shift.hpp"
#include "core/motion/opencv_baselines.hpp"

namespace vl::app {
namespace {

using namespace vl::motion;

// Hue as a fully saturated colour; pixels without a reliable hue in dark grey.
cv::Mat huePicture(const HueImage& h) {
    std::vector<cv::Mat> ch{h.hue, cv::Mat(h.hue.size(), CV_8U, cv::Scalar(255)), cv::Mat(h.hue.size(), CV_8U, cv::Scalar(230))};
    cv::Mat hsv, bgr;
    cv::merge(ch, hsv);
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    bgr.setTo(cv::Scalar(48, 48, 48), h.valid == 0);
    return bgr;
}

class MeanShiftModule final : public Module {
public:
    Info info() const override {
        return {QStringLiteral("track-meanshift"), QStringLiteral("Mean shift"), QStringLiteral("Tracking"),
                QStringLiteral("The object is described by the histogram of its hues. Each pixel of a new frame is "
                               "weighted by how common its hue is in the object (the <i>back-projection</i>), and the "
                               "box moves to the weighted centroid of what it covers, again and again, until it stops: "
                               "it climbs to the nearest blob of object-coloured pixels. A Kalman filter can predict "
                               "where to start, and carry the box through an occlusion."),
                QStringLiteral("on <i>Synthetic: disc behind a pillar</i>, watch the box stick to the pillar; then "
                               "turn on <b>Kalman prediction</b> and see it come out on the other side with the disc."),
                InputKind::Video, 600, QStringLiteral("Frame"), QStringLiteral("cv::meanShift / cv::CamShift"),
                {QStringLiteral("src/core/motion/mean_shift.hpp"), QStringLiteral("src/core/motion/mean_shift.cpp"),
                 QStringLiteral("src/core/motion/kalman.hpp"), QStringLiteral("src/core/motion/kalman.cpp"),
                 QStringLiteral("src/core/motion/tracking.hpp")}};
    }

    std::vector<ParamSpec> params() const override {
        return {
            ParamSpec::integer(QStringLiteral("bins"), QStringLiteral("Hue histogram bins"), 4, 90, 32, 2)
                .withHelp(QStringLiteral("Fewer bins: a coarser colour model that tolerates lighting changes but "
                                         "matches more of the background.")),
            ParamSpec::integer(QStringLiteral("smin"), QStringLiteral("Minimum saturation"), 0, 200, 60, 5)
                .withHelp(QStringLiteral("Greyish pixels have a meaningless hue; below this saturation (of 255) they get "
                                         "no weight.")),
            ParamSpec::real(QStringLiteral("update"), QStringLiteral("Model update rate γ"), 0.0, 0.5, 0.0, 0.01, 2)
                .withHelp(QStringLiteral("After each confident frame, the histogram becomes (1 − γ)·model + γ·current "
                                         "box: follows appearance changes, but can drift onto the background.")),
            ParamSpec::toggle(QStringLiteral("kalman"), QStringLiteral("Kalman prediction"), false)
                .withHelp(QStringLiteral("Start each search from a constant-velocity prediction, and coast on it when the "
                                         "window holds less than 30 % of the model's weight. The dashed ellipse is "
                                         "the 2σ uncertainty of the prediction.")),
            ParamSpec::choice(QStringLiteral("baseline"), QStringLiteral("OpenCV baseline (Compare)"),
                              {QStringLiteral("cv::meanShift"), QStringLiteral("cv::CamShift")}, 0)
                .withHelp(QStringLiteral("What Compare runs on our back-projection from the same box: OpenCV's mean "
                                         "shift (fixed window), or CamShift (the window also adapts its size and angle).")),
        };
    }

    std::vector<Sample> samples() const override { return motion_samples::tracking(); }
    bool wantsRoi() const override { return true; }

    Output compute(const Inputs& in, const Params& p, const RunContext& ctx) const override {
        CV_Assert(!in.images.empty());
        QString note;
        const cv::Rect2d box0 = tracking::startBox(in, p, &note);
        MeanShiftTrackerParams tp;
        tp.hue.bins = intParam(p, "bins");
        tp.hue.minSaturation = intParam(p, "smin");
        tp.modelUpdate = realParam(p, "update");
        tp.usePrediction = toggleParam(p, "kalman");

        MeanShiftTracker tracker(tp, vl::toBGR(in.images.front()), box0);
        Output out;
        tracking::Summary sum;
        sum.hasTruth = motion_samples::truthBox(in, 0).area() > 0;
        std::vector<TrackStep> steps;
        Stopwatch sw;
        // The baseline climbs the same weights, from the same box, without any prediction.
        std::optional<opencv::ShiftTracker> baseline;
        const auto which = static_cast<opencv::Shift>(intParam(p, "baseline"));
        if (ctx.compare) baseline.emplace(which, box0, tp.maxIterations, tp.epsilon);
        tracking::BaselineTrack theirs;
        double oursClimbMs = 0;
        std::vector<double> cvIterations;
        for (size_t t = 0; t < in.images.size(); ++t) {
            ctx.check();
            const cv::Mat frame = vl::toBGR(in.images[t]);
            const MeanShiftTracker::Frame r = t == 0 ? tracker.first() : tracker.step(frame);
            const TrackStep& s = r.step;
            steps.push_back(s);
            sum.score.push_back(100 * s.score);
            sum.iterations.push_back(s.iterations);
            sum.lost += s.lost;
            oursClimbMs += r.climbMs;

            Frame f;
            const cv::Rect2d truth = motion_samples::truthBox(in, t);
            View v0{QStringLiteral("Frame"), toQImage(frame), tracking::trackShapes(steps, truth)};
            // Mechanism: the weights, with the climb of this frame from its start to the final centre.
            View w{QStringLiteral("Weights"), toQImage(vl::colorize(r.weights, 0, 1)), tracking::boxShapes(s), {}, {},
                   Legend{QStringLiteral("weight f_H(hue) / max"), 0, 1, {}, {}}};
            if (s.path.size() > 1) {
                QPolygonF climb;
                for (const auto& q : s.path) climb << QPointF(q.x, q.y);
                w.shapes.push_back(PolylineShape{climb, Pen{colors::accent, 1.5, false}, false});
                w.shapes.push_back(PointsShape{climb, colors::accent, 2.5});
                w.shapes.push_back(LabelShape{QPointF(s.box.br().x + 4, s.box.y), QStringLiteral("%1 steps").arg(s.iterations)});
            }
            View hue{QStringLiteral("Hue"), toQImage(huePicture(r.hue)), tracking::boxShapes(s)};
            cv::Mat w8, hue8 = r.hue.hue, valid = r.hue.valid;
            r.weights.convertTo(w8, CV_8U, 255.0);
            const auto probe = [w8, hue8, valid](QPoint q) {
                if (!valid.at<uchar>(q.y(), q.x())) return QStringLiteral("saturation or value too low: no weight");
                return QStringLiteral("hue %1° · weight %2").arg(2 * hue8.at<uchar>(q.y(), q.x())).arg(w8.at<uchar>(q.y(), q.x()) / 255.0, 0, 'f', 2);
            };
            v0.probe = w.probe = hue.probe = probe;
            f.views = {v0, w, hue};
            if (truth.area() > 0) {
                const double o = iou(s.box, truth);
                sum.iou.push_back(o);
                f.stats.push_back({QStringLiteral("This frame: IoU"), fmt(o, 2)});
            }
            if (baseline) {
                opencv::ShiftTracker::Step b{box0, {}, 0, 0};
                if (t > 0) b = baseline->step(r.weights);
                theirs.boxes.push_back(b.box);
                theirs.ms += b.ms;
                if (t > 0 && b.iterations >= 0) cvIterations.push_back(b.iterations);
                // Same rule as ours: lost when the window holds too little of the first box's weight.
                const double share = tracker.referenceMass() > 0 ? windowMass(r.weights, cv::Rect(b.box)) / tracker.referenceMass() : 0.0;
                const bool lost = t > 0 && share < tp.lostRatio;
                theirs.lost += lost;
                if (truth.area() > 0) theirs.iou.push_back(iou(b.box, truth));
                f.views.push_back(tracking::baselineView(v0.image, theirs, lost, truth, b.fitted));
            }
            f.stats.push_back({QStringLiteral("This frame: model weight"), fmt(100 * s.score, 0) + QStringLiteral(" %")});
            f.stats.push_back({QStringLiteral("This frame: iterations"), QString::number(s.iterations)});
            if (s.lost) f.stats.push_back({QStringLiteral("Status"), QStringLiteral("lost")});
            out.frames.push_back(std::move(f));

            report(ctx.progress, double(t + 1) / double(in.images.size()), "frame " + std::to_string(t + 1));
            if (ctx.partial && motion_samples::partialDue(t)) {
                auto part = std::make_shared<Output>(out);
                tracking::summarise(*part, sum, QStringLiteral("model weight kept"), sw.ms(), true);
                ctx.partial(part);
            }
        }
        tracking::summarise(out, sum, QStringLiteral("model weight kept"), sw.ms(), true);
        if (baseline) {
            tracking::compareRows(out, sum, oursClimbMs, theirs, QStringLiteral("Mean-shift time per frame"));
            if (!cvIterations.empty()) {
                // Frame 0 is the given box for both: count the frames after it.
                double a = 0, b = 0;
                for (size_t t = 1; t < sum.iterations.size(); ++t) a += sum.iterations[t];
                for (double v : cvIterations) b += v;
                out.comparison.push_back({QStringLiteral("Iterations per frame"), a / double(cvIterations.size()),
                                          b / double(cvIterations.size()), {}, 1, true});
            }
        }
        if (note.isEmpty() && sum.lost > int(in.images.size()) / 5)
            note = tp.usePrediction ? QStringLiteral("Lost in more than a fifth of the frames: try fewer bins or a lower saturation threshold.")
                                    : QStringLiteral("Lost in more than a fifth of the frames: try the Kalman prediction.");
        out.note = note;
        return out;
    }
};

}  // namespace

VL_REGISTER_MODULE(MeanShiftModule);

}  // namespace vl::app
