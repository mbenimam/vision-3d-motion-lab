#include "app/module.hpp"

#include <algorithm>
#include <stdexcept>

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QPainter>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace vl::app {

ParamSpec ParamSpec::integer(QString key, QString label, int min, int max, int value, int step, QString suffix) {
    ParamSpec s;
    s.key = std::move(key);
    s.label = std::move(label);
    s.kind = Kind::Int;
    s.min = min;
    s.max = max;
    s.value = value;
    s.step = step;
    s.suffix = std::move(suffix);
    return s;
}

ParamSpec ParamSpec::real(QString key, QString label, double min, double max, double value, double step,
                          int decimals, QString suffix) {
    ParamSpec s;
    s.key = std::move(key);
    s.label = std::move(label);
    s.kind = Kind::Real;
    s.min = min;
    s.max = max;
    s.value = value;
    s.step = step;
    s.decimals = decimals;
    s.suffix = std::move(suffix);
    return s;
}

ParamSpec ParamSpec::choice(QString key, QString label, QStringList choices, int index) {
    ParamSpec s;
    s.key = std::move(key);
    s.label = std::move(label);
    s.kind = Kind::Choice;
    s.choices = std::move(choices);
    s.min = 0;
    s.max = double(s.choices.size() - 1);
    s.value = index;
    return s;
}

ParamSpec ParamSpec::toggle(QString key, QString label, bool on) {
    ParamSpec s;
    s.key = std::move(key);
    s.label = std::move(label);
    s.kind = Kind::Toggle;
    s.value = on ? 1 : 0;
    return s;
}

ParamSpec&& ParamSpec::withHelp(QString text) && {
    help = std::move(text);
    return std::move(*this);
}

int intParam(const Params& p, const QString& key) { return p.value(key).toInt(); }
double realParam(const Params& p, const QString& key) { return p.value(key).toDouble(); }
bool toggleParam(const Params& p, const QString& key) { return p.value(key).toBool(); }
QRectF roiParam(const Params& p) { return p.value(QStringLiteral("roi")).toRectF(); }

namespace {
std::vector<ModuleFactory>& factories() {
    static std::vector<ModuleFactory> f;
    return f;
}
}  // namespace

bool registerModule(ModuleFactory factory) {
    factories().push_back(std::move(factory));
    return true;
}

std::vector<std::unique_ptr<Module>> createModules() {
    std::vector<std::unique_ptr<Module>> out;
    for (const auto& f : factories()) out.push_back(f());
    std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a->info().order < b->info().order; });
    return out;
}

QImage toQImage(const cv::Mat& m) {
    if (m.empty()) return {};
    cv::Mat src = m;
    if (m.depth() != CV_8U) m.convertTo(src, CV_8U);
    switch (src.channels()) {
    case 1:
        return QImage(src.data, src.cols, src.rows, int(src.step), QImage::Format_Grayscale8).copy();
    case 3:
        return QImage(src.data, src.cols, src.rows, int(src.step), QImage::Format_BGR888).copy();
    case 4: {
        cv::Mat rgba;
        cv::cvtColor(src, rgba, cv::COLOR_BGRA2RGBA);
        return QImage(rgba.data, rgba.cols, rgba.rows, int(rgba.step), QImage::Format_RGBA8888).copy();
    }
    default:
        return {};
    }
}

QString fmt(double v, int decimals) { return QString::number(v, 'f', decimals); }

QString fmtMs(double ms) {
    return ms < 1000 ? QStringLiteral("%1 ms").arg(qRound(ms)) : QStringLiteral("%1 s").arg(ms / 1000.0, 0, 'f', 2);
}

std::vector<cv::Mat> loadImages(const QStringList& paths) {
    std::vector<cv::Mat> out;
    for (const auto& p : paths) {
        cv::Mat img = cv::imread(p.toStdString(), cv::IMREAD_COLOR);
        if (img.empty()) throw std::runtime_error("cannot read " + p.toStdString());
        out.push_back(img);
    }
    return out;
}

QString dataFile(const QString& relative) {
    QStringList roots;
    if (const QByteArray env = qgetenv("VL_DATA"); !env.isEmpty()) roots << QString::fromLocal8Bit(env);
    roots << QDir::currentPath() + QStringLiteral("/data");
    const QString exe = QCoreApplication::applicationDirPath();
    roots << exe + QStringLiteral("/data") << exe + QStringLiteral("/../data") << exe + QStringLiteral("/../../data");
#ifdef VL_SOURCE_DIR
    roots << QStringLiteral(VL_SOURCE_DIR "/data");
#endif
    for (const QString& root : roots) {
        const QFileInfo f(root + QLatin1Char('/') + relative);
        if (f.exists()) return f.absoluteFilePath();
    }
    return {};
}

void addSideBySideViews(Output& out) {
    static const QString prefix = QStringLiteral("OpenCV · ");
    for (auto& frame : out.frames) {
        std::vector<View> extra;
        for (const auto& theirs : frame.views) {
            if (!theirs.name.startsWith(prefix)) continue;
            const QString name = theirs.name.mid(prefix.size());
            const auto ours = std::find_if(frame.views.begin(), frame.views.end(), [&](const View& v) { return v.name == name; });
            if (ours == frame.views.end()) continue;
            View pair;
            pair.name = name + QStringLiteral(" · ours | OpenCV");
            // Our overlays stay on the left half; the right half starts after a 12-pixel gap.
            pair.shapes = ours->shapes;
            const View a = *ours, b = theirs;
            pair.render = [a, b] {
                const QImage left = a.picture(), right = b.picture();
                if (left.size() != right.size()) return left;
                const int gap = 12;
                QImage both(left.width() * 2 + gap, left.height(), QImage::Format_RGB888);
                both.fill(QColor(0x0b, 0x0e, 0x14));
                QPainter p(&both);
                p.drawImage(0, 0, left);
                p.drawImage(left.width() + gap, 0, right);
                QFont f = p.font();
                f.setBold(true);
                f.setPixelSize(std::max(11, left.height() / 28));
                p.setFont(f);
                const auto tag = [&](int x, const QString& text) {
                    const QRect r = p.fontMetrics().boundingRect(text).adjusted(-6, -3, 6, 3);
                    p.fillRect(r.translated(x + 8 - r.left(), 8 - r.top()), QColor(14, 17, 23, 210));
                    p.setPen(QColor(0xff, 0xcf, 0x66));
                    p.drawText(x + 14, 8 - r.top() - 3, text);
                };
                tag(0, QObject::tr("ours"));
                tag(left.width() + gap, QObject::tr("OpenCV"));
                return both;
            };
            pair.legend = ours->legend;
            extra.push_back(std::move(pair));
        }
        for (auto& v : extra) frame.views.push_back(std::move(v));
    }
}

}  // namespace vl::app
