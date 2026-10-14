#pragma once
// The contract between the framework and a topic ("module").
//
// A module is a pure description plus a pure function: it says what it needs (an image, a pair,
// a video), which parameters it exposes, and how to turn inputs + parameters into an Output made
// of views (images) with vector overlays. It never touches a widget. The framework owns the UI,
// runs compute() on a worker thread, cancels stale runs and draws the result.

#include <functional>
#include <memory>
#include <optional>
#include <stop_token>
#include <variant>
#include <vector>

#include <QColor>
#include <QImage>
#include <QLineF>
#include <QPolygonF>
#include <QRectF>
#include <QString>
#include <QStringList>
#include <QVariantMap>

#include <opencv2/core.hpp>

#include "core/common/task.hpp"

namespace vl::app {

// ------------------------------------------------------------------ palette shared with the theme
namespace colors {
inline const QColor accent{0xff, 0xb0, 0x00};   // the thing to look at
inline const QColor good{0x4c, 0xd9, 0x7b};     // inliers, accepted
inline const QColor bad{0xff, 0x5c, 0x8a};      // outliers, rejected
inline const QColor info{0x4f, 0xc3, 0xf7};     // secondary series
inline const QColor muted{0x9a, 0xa4, 0xb2};
}  // namespace colors

// ------------------------------------------------------------------ overlays (drawn as vectors, crisp at any zoom)
struct Pen {
    QColor color = colors::accent;
    double width = 1.5;  // in screen pixels (cosmetic)
    bool dashed = false;
};
struct LineShape { QLineF line; Pen pen; };
struct PolylineShape { QPolygonF points; Pen pen; bool closed = false; };
struct PointsShape { QPolygonF points; QColor color = colors::accent; double radius = 3.0; };  // radius in screen px
struct RectShape { QRectF rect; Pen pen; };
struct EllipseShape { QPointF centre; double rx = 0, ry = 0, angleDeg = 0; Pen pen; };
struct ArrowsShape { std::vector<QLineF> arrows; Pen pen; };  // from p1 to p2, with a small head
struct LabelShape { QPointF at; QString text; QColor color = colors::accent; };
using Shape = std::variant<LineShape, PolylineShape, PointsShape, RectShape, EllipseShape, ArrowsShape, LabelShape>;

// ------------------------------------------------------------------ results
// A colour scale drawn in the corner of the canvas: what the colours of a value map mean.
struct Legend {
    QString label;            // "disparity (px)"
    double lo = 0, hi = 1;    // values at the two ends
    QString loText, hiText;   // optional text for the ends (default: the numbers)
    QImage key;               // optional: a picture to show instead of the turbo bar (e.g. a colour wheel)
};

// One picture on the canvas. Coordinates of shapes are image pixels (x right, y down).
struct View {
    QString name;                                // tab label: "Left", "Disparity", ...
    QImage image;                                // use toQImage() to convert a cv::Mat
    std::vector<Shape> shapes;
    std::function<QString(QPoint)> probe;        // optional: text under the cursor ("d = 12.0 px")
    cv::Mat data;                                // optional: the values behind the picture (for onClick)
    std::optional<Legend> legend;                // optional: turbo colour scale (vl::colorize)
    // Optional: draw the picture on demand instead of storing it. Video topics keep compact data per
    // frame and render a view only when it is shown (a 300-frame clip with 5 views would otherwise
    // hold 1500 pictures). Called on the UI thread; must be quick and must not touch shared state.
    std::function<QImage()> render;
    QImage picture() const { return image.isNull() && render ? render() : image; }
};

struct Stat {
    QString label, value;
    bool emphasis = false;  // the one number that matters for this topic
};

struct Series {  // drawn as a sparkline in the inspector (energy per iteration, inliers per trial, ...)
    QString label;
    std::vector<double> values;
};

struct Frame {
    std::vector<View> views;  // every frame has the same view names, in the same order
    std::vector<Stat> stats;  // per-frame numbers (shown under the global ones)
};

// One line of the "ours vs OpenCV" table: the same measurement for our implementation and for
// the OpenCV baseline run on the same input. The inspector highlights the better value.
struct Comparison {
    QString metric;             // "Bad pixels (non-occluded)", "Run time"
    double ours = 0, opencv = 0;
    QString unit;               // " %", " px", " ms"
    int decimals = 1;
    bool lowerIsBetter = true;
};

struct Output {
    std::vector<Frame> frames;  // one frame for image topics, one per video frame otherwise
    std::vector<Stat> stats;
    std::vector<Series> series;
    std::vector<Comparison> comparison;  // filled when RunContext::compare is set
    QString note;               // one line under the result, e.g. a warning ("only 6 matches")
};
using OutputPtr = std::shared_ptr<const Output>;

// ------------------------------------------------------------------ inputs
enum class InputKind { Image, ImagePair, Video };

struct Inputs {
    QString name;
    std::vector<cv::Mat> images;  // 8-bit BGR. Image: 1, ImagePair: 2, Video: the frames
    std::vector<cv::Mat> truth;   // optional ground truth (disparity, flow, masks), module-specific
    double fps = 25.0;
};
using InputsPtr = std::shared_ptr<const Inputs>;

struct Sample {
    QString name;
    std::function<Inputs()> load;  // runs on a worker thread; may be slow (decode, synthesis)
};

// ------------------------------------------------------------------ parameters
struct ParamSpec {
    enum class Kind { Int, Real, Choice, Toggle };
    QString key, label, help;
    Kind kind = Kind::Int;
    double min = 0, max = 1, step = 1, value = 0;  // value: default (choice index, 0/1 for toggles)
    int decimals = 0;
    QString suffix;      // unit shown after the value: " px", " %", "°"
    QStringList choices;

    static ParamSpec integer(QString key, QString label, int min, int max, int value, int step = 1, QString suffix = {});
    static ParamSpec real(QString key, QString label, double min, double max, double value, double step,
                          int decimals, QString suffix = {});
    static ParamSpec choice(QString key, QString label, QStringList choices, int index = 0);
    static ParamSpec toggle(QString key, QString label, bool on);
    ParamSpec&& withHelp(QString text) &&;
};

// Parameter values by key. Choices are stored as their index; the ROI (if any) as a QRectF under "roi".
using Params = QVariantMap;
int intParam(const Params& p, const QString& key);
double realParam(const Params& p, const QString& key);
bool toggleParam(const Params& p, const QString& key);
QRectF roiParam(const Params& p);  // empty when the user has not drawn one

// ------------------------------------------------------------------ running
struct RunContext {
    std::stop_token stop;
    bool compare = false;  // also run the OpenCV baseline: add its views ("OpenCV · ...") and Output::comparison
    ProgressFn progress;                          // fraction + message
    std::function<void(OutputPtr)> partial;       // optional: show intermediate results (throttled by the runner)
    void check() const { checkStop(stop); }
};

class Module {
public:
    struct Info {
        QString id;       // stable key, used for settings and --module
        QString title;    // "Graph cuts"
        QString group;    // "Stereo"
        QString idea;     // 2-3 sentences, rich text allowed (<i>, <b>)
        QString tryHint;  // one concrete thing to do
        InputKind input = InputKind::Image;
        int order = 0;    // position in the navigation, ascending
        QString defaultView;  // view shown first (by name), e.g. the result; empty = the first view
        QString baseline;     // the OpenCV method compared against, e.g. "cv::StereoSGBM"; empty = none
        QStringList sources;  // our implementation, repo-relative ("src/core/stereo/block_matching.cpp"),
                              // shown by the code reader next to the picture
    };

    virtual ~Module() = default;
    virtual Info info() const = 0;
    virtual std::vector<ParamSpec> params() const = 0;
    virtual std::vector<Sample> samples() const = 0;

    // Called on a worker thread. Must be thread-safe (const, no shared mutable state without a lock).
    // Call ctx.check() in loops so a newer request can cancel this one quickly.
    virtual Output compute(const Inputs& in, const Params& p, const RunContext& ctx) const = 0;

    // Called on the UI thread when the user clicks a view (a click, not a drag). Return transient
    // shapes to draw on top of the views, keyed by view index. Must be fast.
    struct ClickShapes { int view = 0; std::vector<Shape> shapes; };
    virtual std::vector<ClickShapes> onClick(const Output&, int /*frame*/, int /*view*/, QPointF /*pixel*/) const {
        return {};
    }

    // Topics that need a region of interest (tracking) return true; the canvas then offers a box tool
    // and the drawn rectangle arrives in params under "roi" (pixel coordinates of the first frame).
    virtual bool wantsRoi() const { return false; }
};

// ------------------------------------------------------------------ registry
using ModuleFactory = std::function<std::unique_ptr<Module>()>;
bool registerModule(ModuleFactory factory);
std::vector<std::unique_ptr<Module>> createModules();  // sorted by Info::order

// Put VL_REGISTER_MODULE(MyModule) at namespace scope in the module's .cpp file.
#define VL_REGISTER_MODULE(Type) \
    static const bool vl_registered_##Type = ::vl::app::registerModule([] { return std::make_unique<Type>(); })

// ------------------------------------------------------------------ helpers for module authors
QImage toQImage(const cv::Mat& bgrOrGray);         // deep copy, 8-bit grey/BGR/BGRA
QString fmt(double v, int decimals = 1);           // "12.3"
QString fmtMs(double ms);                          // "41 ms", "1.20 s"
std::vector<cv::Mat> loadImages(const QStringList& paths);  // BGR, throws on failure
// Absolute path of a file under data/ (filled by scripts/fetch_data.sh), or empty if it is missing.
// Looks in $VL_DATA, ./data, next to the executable and in the source tree.
QString dataFile(const QString& relative);

}  // namespace vl::app
