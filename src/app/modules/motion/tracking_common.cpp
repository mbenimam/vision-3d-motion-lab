#include "app/modules/motion/tracking_common.hpp"

#include <algorithm>

#include "app/modules/motion/motion_samples.hpp"
#include "core/motion/kalman.hpp"

namespace vl::app::tracking {

QRectF toQ(const cv::Rect2d& r) { return {r.x, r.y, r.width, r.height}; }

cv::Rect2d startBox(const Inputs& in, const Params& p, QString* note) {
    const cv::Size size = in.images.front().size();
    const cv::Rect2d frame(0, 0, size.width, size.height);
    QRectF q = roiParam(p);
    if (q.width() < 6 || q.height() < 6) {
        const auto d = motion_samples::defaultBox(in);
        q = d.box;
        if (!d.known && note)
            *note = QStringLiteral("No box drawn yet: tracking a central guess. Use the box tool to draw one around "
                                   "the object on the first frame.");
    }
    cv::Rect2d b = cv::Rect2d(q.x(), q.y(), q.width(), q.height()) & frame;
    if (b.width < 6 || b.height < 6) b = cv::Rect2d(size.width * 0.4, size.height * 0.4, size.width * 0.2, size.height * 0.2);
    return b;
}

std::vector<Shape> boxShapes(const vl::motion::TrackStep& s) {
    std::vector<Shape> out;
    if (s.predicted) {
        const auto e = vl::motion::covarianceEllipse(s.predictionCov, s.prediction, 2.0);
        out.push_back(EllipseShape{{e.centre.x, e.centre.y}, e.major, e.minor, e.angleDeg, Pen{colors::info, 1.2, true}});
        out.push_back(PointsShape{QPolygonF{QPointF(s.prediction.x, s.prediction.y)}, colors::info, 2.5});
    }
    out.push_back(RectShape{toQ(s.box), s.lost ? Pen{colors::bad, 2.0, true} : Pen{colors::accent, 2.0, false}});
    return out;
}

std::vector<Shape> trackShapes(const std::vector<vl::motion::TrackStep>& steps, const cv::Rect2d& truth) {
    std::vector<Shape> out;
    if (truth.area() > 0) out.push_back(RectShape{toQ(truth), Pen{colors::good, 1.2, true}});
    QPolygonF path;
    for (const auto& s : steps) {
        const cv::Point2d c = vl::motion::centreOf(s.box);
        path << QPointF(c.x, c.y);
    }
    if (path.size() > 1) out.push_back(PolylineShape{path, Pen{colors::accent, 1.2, false}, false});
    const auto& cur = steps.back();
    for (auto& s : boxShapes(cur)) out.push_back(std::move(s));
    if (cur.lost)
        out.push_back(LabelShape{QPointF(cur.box.x, cur.box.y - 16), cur.predicted ? QStringLiteral("lost: coasting on the prediction")
                                                                                : QStringLiteral("lost: the model is not here"),
                                  colors::bad});
    return out;
}

void summarise(Output& out, const Summary& s, const QString& scoreLabel, double ms, bool iterations) {
    auto mean = [](const std::vector<double>& v) {
        double t = 0;
        for (double x : v) t += x;
        return v.empty() ? 0.0 : t / double(v.size());
    };
    const size_t n = s.score.size();
    out.stats.clear();
    out.series.clear();
    if (s.hasTruth) {
        out.stats.push_back({QStringLiteral("Mean IoU with the true box"), fmt(mean(s.iou), 3), true});
        out.series.push_back({QStringLiteral("IoU per frame"), s.iou});
    }
    out.stats.push_back({QStringLiteral("Lost frames"), QStringLiteral("%1 of %2").arg(s.lost).arg(n), !s.hasTruth});
    out.stats.push_back({QStringLiteral("Mean ") + scoreLabel, fmt(mean(s.score), 0) + QStringLiteral(" %")});
    if (iterations) {
        out.stats.push_back({QStringLiteral("Mean-shift iterations"), fmt(mean(s.iterations), 1) + QStringLiteral(" per frame")});
        out.series.push_back({QStringLiteral("Iterations, frame by frame"), s.iterations});
    }
    out.stats.push_back({QStringLiteral("Time per frame"), fmtMs(ms / double(std::max<size_t>(1, n)))});
    out.series.push_back({scoreLabel + QStringLiteral(" per frame (%)"), s.score});
}

View baselineView(const QImage& frame, const BaselineTrack& b, bool lost, const cv::Rect2d& truth,
                  const cv::RotatedRect& fitted) {
    View v{QStringLiteral("OpenCV · Frame"), frame};
    if (truth.area() > 0) v.shapes.push_back(RectShape{toQ(truth), Pen{colors::good, 1.2, true}});
    QPolygonF path;
    for (const auto& r : b.boxes) path << QPointF(r.x + 0.5 * r.width, r.y + 0.5 * r.height);
    if (path.size() > 1) v.shapes.push_back(PolylineShape{path, Pen{colors::info, 1.2, false}, false});
    if (fitted.size.area() > 0) {
        cv::Point2f corners[4];
        fitted.points(corners);
        QPolygonF poly;
        for (const auto& c : corners) poly << QPointF(c.x, c.y);
        v.shapes.push_back(PolylineShape{poly, Pen{colors::info, 1.0, true}, true});
    }
    v.shapes.push_back(RectShape{toQ(b.boxes.back()), lost ? Pen{colors::bad, 2.0, true} : Pen{colors::info, 2.0, false}});
    if (lost) v.shapes.push_back(LabelShape{QPointF(b.boxes.back().x, b.boxes.back().y - 16), QStringLiteral("OpenCV: lost"), colors::bad});
    return v;
}

void compareRows(Output& out, const Summary& ours, double oursMs, const BaselineTrack& theirs, const QString& timeLabel) {
    auto mean = [](const std::vector<double>& v) {
        double t = 0;
        for (double x : v) t += x;
        return v.empty() ? 0.0 : t / double(v.size());
    };
    const double n = double(std::max<size_t>(1, ours.score.size()));
    if (ours.hasTruth) out.comparison.push_back({QStringLiteral("Mean IoU with the true box"), mean(ours.iou), mean(theirs.iou), {}, 3, false});
    out.comparison.push_back({QStringLiteral("Lost frames"), double(ours.lost), double(theirs.lost), {}, 0, true});
    out.comparison.push_back({timeLabel, oursMs / n, theirs.ms / n, QStringLiteral(" ms"), 2});
}

}  // namespace vl::app::tracking
