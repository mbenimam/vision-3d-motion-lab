// Topic: tracking by detection with the generalised Hough transform (R-table indexed by gradient
// orientation), optionally restricted to a search window and weighed by a Kalman prediction.

#include <cmath>
#include <numbers>

#include <opencv2/imgproc.hpp>

#include "app/module.hpp"
#include "app/modules/motion/motion_samples.hpp"
#include "app/modules/motion/tracking_common.hpp"
#include "core/common/image.hpp"
#include "core/motion/hough.hpp"

namespace vl::app {
namespace {

using namespace vl::motion;

enum ViewIndex { kFrame = 0, kOrientation = 1, kVotes = 2 };

// Orientation as a hue around the colour wheel; pixels that do not vote in red, like the practical.
cv::Mat orientationPicture(const Orientations& o) {
    cv::Mat h(o.angle.size(), CV_8U);
    for (int y = 0; y < h.rows; ++y) {
        const float* a = o.angle.ptr<float>(y);
        uchar* d = h.ptr<uchar>(y);
        for (int x = 0; x < h.cols; ++x) d[x] = uchar(int((a[x] + std::numbers::pi) / (2 * std::numbers::pi) * 180.0) % 180);
    }
    std::vector<cv::Mat> ch{h, cv::Mat(h.size(), CV_8U, cv::Scalar(200)), cv::Mat(h.size(), CV_8U, cv::Scalar(235))};
    cv::Mat hsv, bgr;
    cv::merge(ch, hsv);
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    bgr.setTo(cv::Scalar(0, 0, 255), o.mask == 0);
    return bgr;
}

// The R-table as an N×3 int matrix (bin, dx, dy), stored with the views so onClick can draw votes.
cv::Mat tableMatrix(const RTable& t) {
    cv::Mat m(int(t.entries), 3, CV_32S);
    int i = 0;
    for (int b = 0; b < t.bins; ++b)
        for (const cv::Point& r : t.offsets[size_t(b)]) {
            m.at<int>(i, 0) = b;
            m.at<int>(i, 1) = r.x;
            m.at<int>(i, 2) = r.y;
            ++i;
        }
    return m;
}

class HoughModule final : public Module {
public:
    Info info() const override {
        return {QStringLiteral("track-hough"), QStringLiteral("Hough transform"), QStringLiteral("Tracking"),
                QStringLiteral("The first box is learnt as an <i>R-table</i>: for each gradient orientation, the "
                               "offsets from the edge pixels that have it to the box centre. In a new frame every "
                               "strong-gradient pixel looks up its orientation and votes for the centres those offsets "
                               "point to; the object's edge pixels all agree on one place, and the arg max of the votes "
                               "is the detection. No colour, and no need to start near the object."),
                QStringLiteral("on <i>Synthetic: orange disc</i>, click a pixel of the Orientation view to see the "
                               "centres it votes for; then raise the gradient threshold to 20 and watch the peak thin."),
                InputKind::Video, 610, QStringLiteral("Frame")};
    }

    std::vector<ParamSpec> params() const override {
        return {
            ParamSpec::real(QStringLiteral("grad"), QStringLiteral("Gradient threshold"), 1.0, 40.0, 4.0, 0.5, 1,
                            QStringLiteral(" grey levels/px"))
                .withHelp(QStringLiteral("Pixels with a weaker gradient (shown red) do not vote: their orientation is "
                                         "mostly noise.")),
            ParamSpec::integer(QStringLiteral("bins"), QStringLiteral("Orientation bins"), 4, 90, 36)
                .withHelp(QStringLiteral("R-table rows over 360°. More bins: more selective votes, but a small rotation "
                                         "of the object moves its pixels to other rows.")),
            ParamSpec::integer(QStringLiteral("radius"), QStringLiteral("Search radius (0 = whole frame)"), 0, 120, 0, 5, QStringLiteral(" px"))
                .withHelp(QStringLiteral("Only accept centres this close to the previous position (or the prediction): "
                                         "faster, and look-alikes elsewhere cannot win.")),
            ParamSpec::toggle(QStringLiteral("kalman"), QStringLiteral("Kalman prediction"), false)
                .withHelp(QStringLiteral("Weigh the vote map by the Gaussian prior of a constant-velocity prediction "
                                         "before taking the arg max, and coast on the prediction when the peak drops "
                                         "below 40 % of its recent typical height.")),
        };
    }

    std::vector<Sample> samples() const override { return motion_samples::tracking(); }
    bool wantsRoi() const override { return true; }

    Output compute(const Inputs& in, const Params& p, const RunContext& ctx) const override {
        CV_Assert(!in.images.empty());
        QString note;
        const cv::Rect2d box0 = tracking::startBox(in, p, &note);
        HoughTrackerParams tp;
        tp.gradient.minMagnitude = float(realParam(p, "grad"));
        tp.gradient.bins = intParam(p, "bins");
        tp.searchRadius = intParam(p, "radius");
        tp.usePrediction = toggleParam(p, "kalman");

        HoughTracker tracker(tp, vl::toBGR(in.images.front()), box0);
        const cv::Mat table = tableMatrix(tracker.table());
        const double entries = double(tracker.table().entries);
        Output out;
        tracking::Summary sum;
        sum.hasTruth = motion_samples::truthBox(in, 0).area() > 0;
        std::vector<TrackStep> steps;
        Stopwatch sw;
        for (size_t t = 0; t < in.images.size(); ++t) {
            ctx.check();
            const cv::Mat frame = vl::toBGR(in.images[t]);
            const HoughTracker::Frame r = t == 0 ? tracker.first() : tracker.step(frame);
            const TrackStep& s = r.step;
            steps.push_back(s);
            sum.score.push_back(100 * s.score);
            sum.lost += s.lost;

            Frame f;
            const cv::Rect2d truth = motion_samples::truthBox(in, t);
            View v0{QStringLiteral("Frame"), toQImage(frame), tracking::trackShapes(steps, truth), {}, table};
            cv::Mat bins = r.orient.bin.clone();
            bins.setTo(0xffff, r.orient.mask == 0);  // non-voting pixels marked, for onClick
            View ori{QStringLiteral("Orientation"), toQImage(orientationPicture(r.orient)), tracking::boxShapes(s), {}, bins};
            if (t == 0) ori.shapes.push_back(RectShape{tracking::toQ(box0), Pen{colors::accent, 2.0, false}});
            double vmax = 0;
            cv::minMaxLoc(r.votes, nullptr, &vmax);
            const double hi = std::max(1.0, vmax);
            View votes{QStringLiteral("Votes"), toQImage(vl::colorize(r.votes, 0, hi)), tracking::boxShapes(s), {}, {},
                       Legend{QStringLiteral("votes for the centre"), 0, hi, {}, {}}};
            if (tp.searchRadius > 0) votes.shapes.push_back(RectShape{tracking::toQ(r.search), Pen{colors::info, 1.0, true}});
            const cv::Point2d det = s.path.back();
            votes.shapes.push_back(PointsShape{QPolygonF{QPointF(det.x, det.y)}, colors::accent, 3.5});
            votes.shapes.push_back(LabelShape{QPointF(det.x + 6, det.y - 18), QStringLiteral("peak %1 votes").arg(r.peak, 0, 'f', 0)});
            cv::Mat v16;
            r.votes.convertTo(v16, CV_16U);
            const cv::Mat mag = r.orient.magnitude, ang = r.orient.angle;
            const auto probe = [v16, mag, ang, thr = tp.gradient.minMagnitude](QPoint q) {
                const float m = mag.at<float>(q.y(), q.x());
                QString text = QStringLiteral("|∇I| = %1").arg(m, 0, 'f', 1);
                text += m >= thr ? QStringLiteral(", θ = %1°").arg(ang.at<float>(q.y(), q.x()) * 180.0 / std::numbers::pi, 0, 'f', 0)
                              : QStringLiteral(" (too weak to vote)");
                return text + QStringLiteral(" · %1 votes here").arg(v16.at<ushort>(q.y(), q.x()));
            };
            v0.probe = ori.probe = votes.probe = probe;
            f.views = {v0, ori, votes};
            if (truth.area() > 0) {
                const double o = iou(s.box, truth);
                sum.iou.push_back(o);
                f.stats.push_back({QStringLiteral("IoU (this frame)"), fmt(o, 2)});
            }
            f.stats.push_back({QStringLiteral("Peak votes"), QStringLiteral("%1 of %2 (%3 %)").arg(r.peak, 0, 'f', 0).arg(entries, 0, 'f', 0).arg(100 * s.score, 0, 'f', 0)});
            if (s.lost) f.stats.push_back({QStringLiteral("Status"), QStringLiteral("lost")});
            out.frames.push_back(std::move(f));

            report(ctx.progress, double(t + 1) / double(in.images.size()), "frame " + std::to_string(t + 1));
            if (ctx.partial && motion_samples::partialDue(t)) {
                auto part = std::make_shared<Output>(out);
                finish(*part, sum, sw.ms(), entries);
                ctx.partial(part);
            }
        }
        finish(out, sum, sw.ms(), entries);
        if (entries < 30)
            note = QStringLiteral("Only %1 model pixels pass the gradient threshold: lower it, or draw a box on a "
                                  "more textured object.").arg(entries);
        if (!note.isEmpty()) out.note = note;
        return out;
    }

    // Click a pixel: the centres it votes for (its orientation's R-table row, shifted to it).
    std::vector<ClickShapes> onClick(const Output& out, int frame, int, QPointF px) const override {
        if (out.frames.empty()) return {};
        const auto& f = out.frames[size_t(std::clamp(frame, 0, int(out.frames.size()) - 1))];
        const cv::Mat& bins = f.views[kOrientation].data;
        const cv::Mat& table = f.views[kFrame].data;
        const int x = int(px.x()), y = int(px.y());
        if (bins.empty() || x < 0 || y < 0 || x >= bins.cols || y >= bins.rows) return {};
        const ushort b = bins.at<ushort>(y, x);
        std::vector<ClickShapes> s{{kOrientation, {}}, {kVotes, {}}};
        const QPointF from(x + 0.5, y + 0.5);
        if (b == 0xffff) {
            s[0].shapes.push_back(LabelShape{from + QPointF(6, -18), QStringLiteral("gradient too weak: no vote"), colors::bad});
            return s;
        }
        QPolygonF targets;
        ArrowsShape arrows{{}, Pen{colors::info, 1.0, false}};
        for (int i = 0; i < table.rows; ++i) {
            if (table.at<int>(i, 0) != b) continue;
            const QPointF to(x + table.at<int>(i, 1) + 0.5, y + table.at<int>(i, 2) + 0.5);
            targets << to;
            if (arrows.arrows.size() < 40) arrows.arrows.emplace_back(from, to);
        }
        for (auto& c : s) {
            c.shapes.push_back(PointsShape{targets, colors::info, 2.0});
            c.shapes.push_back(PointsShape{QPolygonF{from}, colors::accent, 3.5});
        }
        s[0].shapes.push_back(arrows);
        s[0].shapes.push_back(LabelShape{from + QPointF(6, -18), QStringLiteral("orientation bin %1: %2 votes").arg(b).arg(targets.size())});
        return s;
    }

private:
    static void finish(Output& out, const tracking::Summary& sum, double ms, double entries) {
        tracking::summarise(out, sum, QStringLiteral("peak share of the model"), ms, false);
        out.stats.push_back({QStringLiteral("R-table entries"), QString::number(entries, 'f', 0)});
    }
};

}  // namespace

VL_REGISTER_MODULE(HoughModule);

}  // namespace vl::app
