// Topic: a document scanner from scratch: Canny edges, Hough lines, the page's quadrilateral, a
// homography by the DLT, inverse warping, illumination flattening and adaptive thresholding.

#include <algorithm>
#include <cmath>
#include <numbers>

#include <opencv2/imgproc.hpp>

#include "app/module.hpp"
#include "app/modules/scanner/scanner_samples.hpp"
#include "core/common/image.hpp"
#include "core/common/task.hpp"
#include "core/scanner/opencv_baselines.hpp"
#include "core/scanner/scanner.hpp"

namespace vl::app {
namespace {

namespace sc = vl::scanner;

const char* const kCornerNames[4] = {"TL", "TR", "BR", "BL"};

// The part of the line ρ = x cos θ + y sin θ inside a w × h picture (empty when it misses it).
QLineF clipToImage(double theta, double rho, double w, double h) {
    const double c = std::cos(theta), s = std::sin(theta);
    std::vector<QPointF> hits;
    if (std::abs(s) > 1e-9)
        for (double x : {0.0, w}) {
            const double y = (rho - x * c) / s;
            if (y >= 0 && y <= h) hits.emplace_back(x, y);
        }
    if (std::abs(c) > 1e-9)
        for (double y : {0.0, h}) {
            const double x = (rho - y * s) / c;
            if (x >= 0 && x <= w) hits.emplace_back(x, y);
        }
    if (hits.size() < 2) return {};
    return {hits.front(), hits.back()};
}

QPolygonF toPolygon(const sc::Quad& q) {
    QPolygonF p;
    for (const auto& c : q) p << QPointF(c.x, c.y);
    return p;
}

cv::Mat dimmed(const cv::Mat& bgr, double keep) {
    cv::Mat d;
    bgr.convertTo(d, -1, keep, 0);
    return d;
}

int viewIndex(const Frame& f, const QString& name) {
    for (int i = 0; i < int(f.views.size()); ++i)
        if (f.views[size_t(i)].name == name) return i;
    return -1;
}

// The photo with a quadrilateral, its corner names, and (when known) the true corners.
View cornersView(const QString& name, const cv::Mat& photo, const sc::Quad& q, const std::optional<sc::Quad>& truth,
                 bool draggable, QColor colour) {
    View v{name, toQImage(photo)};
    // The outline is drawn as a shape too, so it also shows in the side-by-side view (which has no handles).
    v.shapes.push_back(PolylineShape{toPolygon(q), {colour, 2.5}, true});
    if (draggable)
        for (const auto& c : q) v.handles.emplace_back(c.x, c.y);  // rings the user can drag
    else
        v.shapes.push_back(PointsShape{toPolygon(q), colour, 5});
    const cv::Point2d centre = (q[0] + q[1] + q[2] + q[3]) * 0.25;
    for (size_t i = 0; i < 4; ++i) {
        // Labels sit outside the page, along the line from its centre through the corner.
        cv::Point2d out = q[i] - centre;
        out *= 1.0 / std::max(1.0, std::hypot(out.x, out.y));
        v.shapes.push_back(LabelShape{QPointF(q[i].x + 40 * out.x - 10, q[i].y + 40 * out.y - 9),
                                      QString::fromLatin1(kCornerNames[i]), colour});
    }
    if (truth) v.shapes.push_back(PointsShape{toPolygon(*truth), colors::good, 3});
    return v;
}

class ScannerModule final : public Module {
public:
    Info info() const override {
        return {
            QStringLiteral("doc-scanner"),
            QStringLiteral("Document scanner"),
            QStringLiteral("Applications"),
            QStringLiteral("The page's four sides are the longest straight edges of the photo: <b>Canny</b> finds "
                           "thin edges, every edge pixel votes in a <b>Hough</b> accumulator for the lines through "
                           "it, and the best convex quadrilateral of strong lines is the page. Its corners and an "
                           "upright rectangle fix a <b>homography</b> (the DLT); each pixel of the flat page is read "
                           "from the photo through H⁻¹, then the light is divided out."),
            QStringLiteral("open <i>Synthetic: cluttered, corner hidden</i> and turn on Compare: the mug hides a "
                           "corner, the contour recipe fails and the line intersections still land on it. Then drag "
                           "a corner on <i>Corners</i>."),
            InputKind::Image,
            900,
            QStringLiteral("Scan"),
            QStringLiteral("cv::findContours + approxPolyDP"),
            {QStringLiteral("src/core/scanner/scanner.hpp"), QStringLiteral("src/core/scanner/scanner.cpp"),
             QStringLiteral("src/core/scanner/edges.hpp"), QStringLiteral("src/core/scanner/edges.cpp"),
             QStringLiteral("src/core/scanner/hough.hpp"), QStringLiteral("src/core/scanner/hough.cpp"),
             QStringLiteral("src/core/scanner/quad.hpp"), QStringLiteral("src/core/scanner/quad.cpp"),
             QStringLiteral("src/core/scanner/rectify.hpp"), QStringLiteral("src/core/scanner/rectify.cpp"),
             QStringLiteral("src/core/scanner/cleanup.hpp"), QStringLiteral("src/core/scanner/cleanup.cpp")}};
    }

    std::vector<ParamSpec> params() const override {
        return {
            ParamSpec::choice(QStringLiteral("thresholds"), QStringLiteral("Edge thresholds"), {"Automatic", "Manual"},
                              0)
                .withHelp(QStringLiteral("Automatic: T_high is the 92nd percentile of the thinned gradient strengths, "
                                         "so it follows the photo's contrast; T_low = 0.4 T_high.")),
            ParamSpec::real(QStringLiteral("high"), QStringLiteral("High threshold (manual)"), 1, 40, 12, 0.5, 1,
                            QStringLiteral(" grey/px"))
                .withHelp(QStringLiteral("Gradient strength an edge must reach somewhere to be kept (Manual only).")),
            ParamSpec::toggle(QStringLiteral("refine"), QStringLiteral("Sub-pixel side refinement"), true)
                .withHelp(QStringLiteral("Re-fit each side on the full-size photo. Off: the corners come from the "
                                         "half-size Hough lines, 1° and 1 px bins.")),
            ParamSpec::choice(QStringLiteral("ratio"), QStringLiteral("Page shape"), {"From the sides", "A4 (1 : √2)"},
                              0)
                .withHelp(QStringLiteral(
                    "Output size from the longer of each pair of opposite sides, or the ISO paper ratio.")),
            ParamSpec::choice(QStringLiteral("mode"), QStringLiteral("Output"), {"Colour", "Grey", "Black and white"},
                              0)
                .withHelp(QStringLiteral("Black & white adds the adaptive threshold: ink where the page is darker than "
                                         "its neighbourhood's mean minus C.")),
            ParamSpec::integer(QStringLiteral("shadow"), QStringLiteral("Shadow removal"), 0, 100, 100, 5,
                               QStringLiteral(" %"))
                .withHelp(QStringLiteral("How much of the estimated light is divided out. 0 %: a plain contrast "
                                         "stretch, so the shadow stays.")),
            ParamSpec::integer(QStringLiteral("window"), QStringLiteral("Threshold window radius"), 3, 40, 12, 1,
                               QStringLiteral(" px"))
                .withHelp(QStringLiteral("Black & white: the local mean is taken over a (2r+1)² square of the page.")),
            ParamSpec::real(QStringLiteral("C"), QStringLiteral("Threshold offset C"), 0, 40, 10, 1, 0,
                            QStringLiteral(" grey levels"))
                .withHelp(
                    QStringLiteral("Black & white: how much darker than the local mean a pixel must be to be ink.")),
            ParamSpec::toggle(QStringLiteral("dragged"), QStringLiteral("Use dragged corners"), true)
                .withHelp(QStringLiteral("Drag the rings on Corners to fix a detection. Switch off (or Reset) to "
                                         "detect the page again.")),
        };
    }

    std::vector<Sample> samples() const override { return scanner_samples::all(); }

    Output compute(const Inputs& in, const Params& p, const RunContext& ctx) const override {
        const cv::Mat photo = vl::toBGR(in.images[0]);
        const sc::ScanParams sp = scanParams(p);
        const auto truth = scanner_samples::trueCorners(in);

        // Detection always runs, so Edges and Lines show what the method sees even after a drag.
        const vl::Stopwatch clock;
        vl::report(ctx.progress, 0.1, "finding the page");
        const sc::PageDetection det = sc::findPage(photo, sp, ctx.stop);
        const auto dragged = draggedCorners(p, photo.size());
        vl::report(ctx.progress, 0.6, "rectifying");
        const sc::ScanResult r =
            sc::scanDocument(photo, sp, dragged ? dragged : std::optional<sc::Quad>(det.corners), ctx.stop);
        const double oursMs = clock.ms();

        Frame f;
        f.views.push_back(edgesView(photo, det));
        f.views.push_back(houghView(det));
        f.views.push_back(linesView(photo, det));
        f.views.push_back(cornersView(QStringLiteral("Corners"), photo, r.corners, truth, true, colors::accent));
        f.views.push_back(View{QStringLiteral("Flat"), toQImage(r.rectified)});
        f.views.push_back(View{QStringLiteral("Light"), toQImage(r.clean.illumination)});
        f.views.push_back(View{QStringLiteral("Scan"), toQImage(r.clean.scan)});

        Output out;
        addStats(in, det, r, dragged.has_value(), out);
        if (ctx.compare) compareWithOpenCv(in, photo, sp, det, r, oursMs, f, out);
        out.frames.push_back(std::move(f));
        return out;
    }

    // Click the Hough space: the line of that (θ, ρ) bin is drawn on Edges and Lines.
    std::vector<ClickShapes> onClick(const Output& out, int, int view, QPointF px) const override {
        if (out.frames.empty()) return {};
        const Frame& f = out.frames.front();
        const int hough = viewIndex(f, QStringLiteral("Hough"));
        if (view != hough || hough < 0) return {};
        const cv::Mat& meta = f.views[size_t(hough)].data;  // votes, then the scale in the last row's first cell
        const int theta = int(px.x()) / kThetaZoom, row = int(px.y());
        if (theta < 0 || theta >= 180 || row < 0 || row >= meta.rows - 1) return {};
        const double scale = meta.at<int>(meta.rows - 1, 0), offset = (meta.rows - 2) / 2;
        const QImage& photo = f.views.front().image;
        const QLineF line =
            clipToImage(theta * std::numbers::pi / 180, (row - offset) * scale, photo.width(), photo.height());
        std::vector<ClickShapes> s;
        for (const QString& name : {QStringLiteral("Edges"), QStringLiteral("Lines")})
            if (const int v = viewIndex(f, name); v >= 0 && !line.isNull())
                s.push_back(
                    {v,
                     {LineShape{line, {colors::bad, 2.0, false}},
                      LabelShape{line.center(),
                                 QStringLiteral("θ = %1°, ρ = %2 px").arg(theta).arg((row - offset) * scale, 0, 'f', 0),
                                 colors::bad}}});
        s.push_back(
            {hough,
             {PointsShape{QPolygonF{QPointF(theta * kThetaZoom + kThetaZoom / 2.0, row + 0.5)}, colors::bad, 6}}});
        return s;
    }

private:
    static constexpr int kThetaZoom = 6;  // Hough space drawn 6 px per degree

    static sc::ScanParams scanParams(const Params& p) {
        sc::ScanParams sp;
        sp.canny.automatic = intParam(p, "thresholds") == 0;
        sp.canny.high = realParam(p, "high");
        sp.refineSides = toggleParam(p, "refine");
        sp.ratio = intParam(p, "ratio") == 1 ? sc::PageRatio::A4 : sc::PageRatio::Auto;
        sp.cleanup.mode = static_cast<sc::ScanMode>(intParam(p, "mode"));
        sp.cleanup.shadowRemoval = intParam(p, "shadow") / 100.0;
        sp.cleanup.window = intParam(p, "window");
        sp.cleanup.C = realParam(p, "C");
        return sp;
    }

    // The four handles the user dragged, if any, if wanted, and if they fit this photo.
    static std::optional<sc::Quad> draggedCorners(const Params& p, cv::Size size) {
        const auto h = handlesParam(p);
        if (!toggleParam(p, "dragged") || h.size() != 4) return std::nullopt;
        sc::Quad q;
        for (size_t i = 0; i < 4; ++i) {
            if (h[i].x() < 0 || h[i].y() < 0 || h[i].x() > size.width || h[i].y() > size.height) return std::nullopt;
            q[i] = {h[i].x(), h[i].y()};
        }
        return q;
    }

    static View edgesView(const cv::Mat& photo, const sc::PageDetection& det) {
        cv::Mat big;
        cv::resize(det.edges.edges, big, photo.size(), 0, 0, cv::INTER_NEAREST);  // display only
        View v{QStringLiteral("Edges"), toQImage(big)};
        const cv::Mat magnitude = det.edges.magnitude;
        const double s = det.scale;
        v.probe = [magnitude, s](QPoint q) {
            const int x = std::clamp(int(q.x() / s), 0, magnitude.cols - 1),
                      y = std::clamp(int(q.y() / s), 0, magnitude.rows - 1);
            return QStringLiteral("‖∇I‖ = %1 grey/px").arg(magnitude.at<float>(y, x), 0, 'f', 1);
        };
        v.shapes.push_back(LabelShape{
            QPointF(12, 10),
            QStringLiteral("T_low %1 · T_high %2 grey/px").arg(det.edges.low, 0, 'f', 1).arg(det.edges.high, 0, 'f', 1),
            colors::info});
        return v;
    }

    // The accumulator, √votes for contrast, 6 px per degree; peaks in blue, the page's four in amber.
    static View houghView(const sc::PageDetection& det) {
        const cv::Mat& A = det.hough.votes;
        double hi = 1;
        cv::minMaxLoc(A, nullptr, &hi);
        cv::Mat root;
        A.convertTo(root, CV_32F);
        cv::sqrt(root, root);
        cv::Mat shown;
        cv::resize(vl::colorize(root, 0, std::sqrt(hi)), shown, {A.cols * kThetaZoom, A.rows}, 0, 0, cv::INTER_NEAREST);
        // View::data: the votes, plus one row holding the detection scale (for onClick).
        cv::Mat meta = cv::Mat::zeros(A.rows + 1, A.cols, CV_32S);
        A.copyTo(meta.rowRange(0, A.rows));
        meta.at<int>(A.rows, 0) = int(det.scale);
        View v{QStringLiteral("Hough"),
               toQImage(shown),
               {},
               {},
               meta,
               Legend{QStringLiteral("votes (√ scale)"), 0, hi, {}, {}}};
        const int offset = det.hough.rhoOffset;
        v.probe = [A, offset](QPoint q) {
            const int t = q.x() / kThetaZoom, r = q.y();
            if (t < 0 || t >= A.cols || r < 0 || r >= A.rows) return QString();
            return QStringLiteral("θ = %1°  ρ = %2 px  ·  %3 votes  ·  click to see the line")
                .arg(t)
                .arg(r - offset)
                .arg(A.at<int>(r, t));
        };
        // Axis marks: θ along the top, ρ = 0 across the middle.
        for (int t : {0, 45, 90, 135})
            v.shapes.push_back(
                LabelShape{QPointF(t * kThetaZoom + 4, 6), QStringLiteral("θ = %1°").arg(t), colors::muted});
        v.shapes.push_back(
            LineShape{QLineF(0, offset + 0.5, A.cols * kThetaZoom, offset + 0.5), {colors::muted, 1.0, true}});
        v.shapes.push_back(LabelShape{QPointF(4, offset + 4), QStringLiteral("ρ = 0"), colors::muted});
        for (size_t i = 0; i < det.hough.lines.size(); ++i) {
            const auto& l = det.hough.lines[i];
            const bool chosen =
                det.quad && std::find(det.quad->lines.begin(), det.quad->lines.end(), i) != det.quad->lines.end();
            const QPointF at(l.theta * 180 / std::numbers::pi * kThetaZoom + kThetaZoom / 2.0, l.rho + offset + 0.5);
            v.shapes.push_back(EllipseShape{at, 9, 9, 0, {chosen ? colors::accent : colors::info, chosen ? 2.5 : 1.2}});
        }
        return v;
    }

    // Every peak as a line across the photo (blue), the chosen quadrilateral in amber.
    static View linesView(const cv::Mat& photo, const sc::PageDetection& det) {
        View v{QStringLiteral("Lines"), toQImage(dimmed(photo, 0.45))};
        for (const auto& l : det.hough.lines) {
            const QLineF seg = clipToImage(l.theta, l.rho * det.scale, photo.cols, photo.rows);
            if (!seg.isNull()) v.shapes.push_back(LineShape{seg, {colors::info, 1.0, true}});
        }
        if (det.found) {
            v.shapes.push_back(PolylineShape{toPolygon(det.corners), {colors::accent, 2.5}, true});
            const auto& s = det.quad->support;
            const double weakest = *std::min_element(s.begin(), s.end());
            v.shapes.push_back(
                LabelShape{QPointF(12, 10),
                           QStringLiteral("%1 lines · page = largest well-supported quadrilateral (weakest side %2 %)")
                               .arg(det.hough.lines.size())
                               .arg(100 * weakest, 0, 'f', 0)});
        }
        return v;
    }

    static void addStats(const Inputs& in, const sc::PageDetection& det, const sc::ScanResult& r, bool dragged,
                         Output& out) {
        const auto truth = scanner_samples::trueCorners(in);
        const cv::Mat page = scanner_samples::truePage(in);
        if (truth) {
            out.stats.push_back({QStringLiteral("Corner error"),
                                 fmt(sc::cornerError(r.corners, *truth), 2) + QStringLiteral(" px"), true});
            const cv::Mat exact = sc::rectify(in.images[0], *truth, page.size());
            out.stats.push_back(
                {QStringLiteral("Rectification PSNR (vs exact unwarp)"),
                 fmt(sc::psnr(sc::rectify(in.images[0], r.corners, page.size()), exact)) + QStringLiteral(" dB")});
        } else if (det.found) {
            const auto& s = det.quad->support;
            out.stats.push_back({QStringLiteral("Weakest side on edges"),
                                 fmt(100 * *std::min_element(s.begin(), s.end()), 0) + QStringLiteral(" %"), true});
        }
        out.stats.push_back({QStringLiteral("Corners"), dragged     ? QStringLiteral("dragged by you")
                                                        : det.found ? QStringLiteral("detected")
                                                                    : QStringLiteral("not found")});
        out.stats.push_back(
            {QStringLiteral("Flat page"), QStringLiteral("%1 × %2 px").arg(r.size.width).arg(r.size.height)});
        out.stats.push_back({QStringLiteral("Hough lines kept"), QString::number(det.hough.lines.size())});
        if (dragged)
            out.note = QStringLiteral(
                "Using the corners you dragged. Switch off “Use dragged corners” or press Reset to detect again.");
        else if (!det.found)
            out.note = QStringLiteral("No quadrilateral of strong edges found: drag the four corners onto the page.");
    }

    // The classic OpenCV recipe on the same photo, with our Canny thresholds and our output size.
    static void compareWithOpenCv(const Inputs& in, const cv::Mat& photo, const sc::ScanParams& sp,
                                  const sc::PageDetection& det, const sc::ScanResult& r, double oursMs, Frame& f,
                                  Output& out) {
        const auto o =
            sc::opencv::documentScan(photo, sp.detectionSide, det.edges.low, det.edges.high, r.size, sp.cleanup);
        const auto truth = scanner_samples::trueCorners(in);
        f.views.push_back(
            cornersView(QStringLiteral("OpenCV · Corners"), photo, o.corners, truth, false, colors::info));
        f.views.push_back(View{QStringLiteral("OpenCV · Scan"), toQImage(o.scan)});
        if (truth) {
            const cv::Mat page = scanner_samples::truePage(in);
            const cv::Mat exact = sc::rectify(photo, *truth, page.size());
            out.comparison.push_back({QStringLiteral("Corner error"), sc::cornerError(r.corners, *truth),
                                      sc::cornerError(o.corners, *truth), QStringLiteral(" px"), 2});
            out.comparison.push_back(
                {QStringLiteral("Rectification PSNR"), sc::psnr(sc::rectify(photo, r.corners, page.size()), exact),
                 sc::psnr(sc::opencv::rectify(photo, o.corners, page.size()), exact), QStringLiteral(" dB"), 1, false});
        }
        out.comparison.push_back(
            {QStringLiteral("Page found"), det.found ? 1.0 : 0.0, o.found ? 1.0 : 0.0, QString(), 0, false});
        out.comparison.push_back({QStringLiteral("Run time"), oursMs, o.ms, QStringLiteral(" ms"), 0});
        if (!o.found)
            out.note =
                QStringLiteral("OpenCV's recipe found no 4-vertex contour: its corners fall back to the whole photo.");
    }
};

}  // namespace

VL_REGISTER_MODULE(ScannerModule);

}  // namespace vl::app
