#include "app/controller/app_controller.hpp"

#include <QFileDialog>
#include <QFileInfo>
#include <QSettings>
#include <QStandardPaths>
#include <QTimer>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "app/view/canvas.hpp"
#include "app/view/code_reader.hpp"
#include "app/view/inspector.hpp"
#include "app/view/main_window.hpp"
#include "app/view/nav_list.hpp"
#include "app/view/timeline.hpp"
#include "core/common/image.hpp"

namespace vl::app {
namespace {

constexpr int kParamDelayMs = 120;   // a slider drag becomes one run after 120 ms of quiet
constexpr int kMaxSide = 720;        // user pictures are reduced so every topic stays interactive
constexpr int kMaxFrames = 400;

Inputs loadVideo(const QString& path) {
    cv::VideoCapture cap(path.toStdString());
    if (!cap.isOpened()) throw std::runtime_error("cannot open video " + path.toStdString());
    Inputs in;
    in.name = QFileInfo(path).fileName();
    in.fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 25.0;
    cv::Mat frame;
    while (int(in.images.size()) < kMaxFrames && cap.read(frame)) in.images.push_back(vl::limitSize(frame, 480).clone());
    if (in.images.empty()) throw std::runtime_error("no frames in " + path.toStdString());
    return in;
}

}  // namespace

AppController::AppController(QObject* parent) : QObject(parent), doc_(std::make_unique<Document>(createModules())) {
    window_ = std::make_unique<MainWindow>(*doc_);
    auto* w = window_.get();

    // ---- view -> controller
    connect(w->nav(), &NavList::moduleChosen, this, &AppController::chooseModule);
    connect(w->inspector(), &Inspector::paramChanged, this, &AppController::setParam);
    connect(w->inspector(), &Inspector::resetRequested, this, [this] {
        doc_->resetParams();
        window_->inspector()->syncParams(doc_->current().params);
        recompute(0);
    });
    connect(w->inspector(), &Inspector::sampleChosen, this, &AppController::selectSample);
    connect(w->inspector(), &Inspector::openRequested, this, &AppController::openFiles);
    connect(w, &MainWindow::openRequested, this, &AppController::openFiles);
    connect(w->inspector(), &Inspector::boxToolToggled, w->canvas(), &Canvas::setBoxTool);
    connect(w, &MainWindow::viewChosen, this, &AppController::setView);
    connect(w, &MainWindow::exportRequested, this, &AppController::exportPicture);
    connect(w, &MainWindow::compareToggled, this, &AppController::setCompare);
    connect(w, &MainWindow::codeRequested, this, &AppController::showCode);
    connect(w->timeline(), &Timeline::frameChosen, this, &AppController::setFrame);
    connect(w->canvas(), &Canvas::clicked, this, &AppController::click);
    connect(w->canvas(), &Canvas::handlesMoved, this, [this](const std::vector<QPointF>& handles) {
        QVariantList list;
        for (const QPointF& h : handles) list.push_back(h);
        setParam(QStringLiteral("handles"), list);
    });
    connect(w->canvas(), &Canvas::boxDrawn, this, [this](QRectF r) {
        window_->canvas()->setBoxTool(false);
        window_->inspector()->setBoxToolChecked(false);
        doc_->setFrame(0);
        setParam(QStringLiteral("roi"), r);
    });

    // ---- worker -> model -> view
    connect(&runner_, &Runner::started, this, [this] { window_->setBusy(true); });
    connect(&runner_, &Runner::progress, w, &MainWindow::setProgress);
    connect(&runner_, &Runner::partial, this, [this](OutputPtr out) {
        doc_->setPartialOutput(std::move(out));
        window_->showOutput(true);
    });
    connect(&runner_, &Runner::finished, this, [this](OutputPtr out, double ms) {
        doc_->setOutput(std::move(out), ms);
        refreshClickShapes();
        window_->showOutput(true);
    });
    connect(&runner_, &Runner::failed, this, [this](const QString& msg) {
        doc_->setError(msg);
        window_->showError(msg);
    });
    connect(&runner_, &Runner::idle, this, [this] {
        window_->setBusy(false);
        if (onIdle_ && !loading_) {
            auto fn = std::move(onIdle_);
            onIdle_ = {};
            QTimer::singleShot(50, this, fn);
        }
    });
    connect(&runner_, &Runner::loaded, this, [this](InputsPtr in) {
        loading_ = false;
        doc_->setInputs(std::move(in));
        window_->inspector()->setSampleIndex(doc_->current().sampleIndex);
        lastClick_ = {-1, -1};
        window_->showOutput(false);
        recompute(0);
    });
    connect(&runner_, &Runner::loadFailed, this, [this](const QString& msg) {
        loading_ = false;
        window_->setBusy(false);
        doc_->setError(tr("Could not load the input: %1").arg(msg));
        window_->showError(doc_->current().error);
    });
}

AppController::~AppController() { runner_.cancel(); }

void AppController::start(const QString& moduleId) {
    QSettings settings;
    int index = doc_->indexOf(moduleId.isEmpty() ? settings.value(QStringLiteral("module")).toString() : moduleId);
    if (index < 0) index = 0;
    window_->show();
    // setCurrent ignores "no change", so force the first display explicitly.
    if (index == doc_->currentIndex()) {
        window_->showModule();
        loadSample();
    } else {
        chooseModule(index);
    }
}

void AppController::chooseModule(int index) {
    if (index < 0 || index >= doc_->count()) return;
    runner_.cancel();  // the previous topic's run is no longer wanted
    window_->setBusy(false);
    doc_->setCurrent(index);
    QSettings().setValue(QStringLiteral("module"), doc_->current().info.id);
    window_->showModule();
    const auto& s = doc_->current();
    if (!s.inputs) {
        loadSample();
    } else {
        window_->showOutput(false);
        if (!s.output) recompute(0);
    }
}

void AppController::selectSample(int index) {
    doc_->setSampleIndex(index);
    loadSample();
}

void AppController::loadSample() {
    const auto& s = doc_->current();
    if (s.samples.empty()) return;
    runner_.cancel();
    loading_ = true;
    window_->setBusy(true);
    window_->setProgress(0.0, tr("Loading %1…").arg(s.samples[size_t(std::clamp(s.sampleIndex, 0, int(s.samples.size()) - 1))].name));
    runner_.load(s.samples[size_t(std::clamp(s.sampleIndex, 0, int(s.samples.size()) - 1))].load);
}

void AppController::openFiles() {
    const auto& s = doc_->current();
    const QString dir = QSettings().value(QStringLiteral("lastDir"),
                                          QStandardPaths::writableLocation(QStandardPaths::PicturesLocation)).toString();
    const QString images = tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.ppm *.pgm)");
    QStringList files;
    switch (s.info.input) {
    case InputKind::Image:
        files = QStringList{QFileDialog::getOpenFileName(window_.get(), tr("Open an image"), dir, images)};
        break;
    case InputKind::ImagePair:
        files = QFileDialog::getOpenFileNames(window_.get(), tr("Open two images (left, then right)"), dir, images);
        if (files.size() == 1)
            files << QFileDialog::getOpenFileName(window_.get(), tr("Open the second image"), dir, images);
        break;
    case InputKind::Video:
        files = QFileDialog::getOpenFileNames(window_.get(), tr("Open a video, or a sequence of images"), dir,
                                              tr("Video or images (*.mp4 *.avi *.mov *.mkv *.webm *.png *.jpg *.jpeg *.bmp)"));
        break;
    }
    files.removeAll(QString());
    if (files.isEmpty()) return;
    QSettings().setValue(QStringLiteral("lastDir"), QFileInfo(files.front()).absolutePath());
    if (s.info.input == InputKind::ImagePair && files.size() < 2) {
        doc_->setError(tr("Stereo topics need two pictures of the same scene."));
        window_->showError(doc_->current().error);
        return;
    }
    const InputKind kind = s.info.input;
    runner_.cancel();
    loading_ = true;
    window_->setBusy(true);
    runner_.load([files, kind]() -> Inputs {
        const bool isVideo = kind == InputKind::Video && files.size() == 1 &&
                             !QStringList{"png", "jpg", "jpeg", "bmp"}.contains(QFileInfo(files.front()).suffix().toLower());
        if (isVideo) return loadVideo(files.front());
        Inputs in;
        in.name = QFileInfo(files.front()).fileName();
        QStringList sorted = files;
        if (kind == InputKind::Video) sorted.sort();
        for (auto& img : loadImages(kind == InputKind::ImagePair ? sorted.mid(0, 2) : sorted))
            in.images.push_back(vl::limitSize(img, kind == InputKind::Video ? 480 : kMaxSide).clone());
        if (kind == InputKind::ImagePair && in.images[0].size() != in.images[1].size())
            cv::resize(in.images[1], in.images[1], in.images[0].size(), 0, 0, cv::INTER_AREA);
        return in;
    });
}

void AppController::setParam(const QString& key, const QVariant& value) {
    doc_->setParam(key, value);
    recompute(kParamDelayMs);
}

void AppController::recompute(int delayMs) {
    const auto& s = doc_->current();
    if (!s.inputs) return;
    runner_.submit(s.module.get(), s.inputs, s.params, doc_->compare() && !s.info.baseline.isEmpty(), delayMs);
}

void AppController::setView(int view) {
    doc_->setView(view);
    window_->showOutput(true);
}

void AppController::setFrame(int frame) {
    doc_->setFrame(frame);
    window_->showOutput(true);
}

void AppController::click(QPointF pixel) {
    lastClick_ = pixel;
    refreshClickShapes();
    window_->showOutput(true);
}

// Clicked overlays are derived from the result, so they are recomputed when the result changes
// (move a slider and the epipolar line of the point you clicked follows the new estimate).
void AppController::refreshClickShapes() {
    const auto& s = doc_->current();
    if (!s.output || lastClick_.x() < 0) return;
    if (s.view == 0) {  // the clean Input view takes no overlays
        doc_->setClickShapes({});
        return;
    }
    // Modules index their own views; the framework put "Input" in front of them. Hand the module its
    // own output and shift the views of the shapes it returns.
    Output own = *s.output;
    for (auto& f : own.frames)
        if (!f.views.empty()) f.views.erase(f.views.begin());
    auto shapes = s.module->onClick(own, s.frame, s.view - 1, lastClick_);
    for (auto& cs : shapes) ++cs.view;
    doc_->setClickShapes(std::move(shapes));
}

void AppController::exportPicture() {
    const QImage img = window_->canvas()->renderCurrent();
    if (img.isNull()) return;
    const auto& s = doc_->current();
    const QString suggested = QSettings().value(QStringLiteral("lastDir")).toString() + QStringLiteral("/") +
                              s.info.id + QStringLiteral(".png");
    const QString path = QFileDialog::getSaveFileName(window_.get(), tr("Export picture"), suggested, tr("PNG (*.png)"));
    if (!path.isEmpty()) img.save(path);
}

void AppController::setCompare(bool on) {
    doc_->setCompare(on);
    window_->showOutput(true);
    recompute(0);
}

void AppController::showCode() {
    if (!codeReader_) codeReader_ = new CodeReader(window_.get());
    const auto& info = doc_->current().info;
    codeReader_->showTopic(info.title, info.sources);
    codeReader_->show();
    codeReader_->raise();
}

void AppController::syncInspector() { window_->inspector()->syncParams(doc_->current().params); }

void AppController::whenIdle(std::function<void()> fn) { onIdle_ = std::move(fn); }

}  // namespace vl::app
