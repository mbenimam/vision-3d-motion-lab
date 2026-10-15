#include "app/view/main_window.hpp"

#include <QButtonGroup>
#include <QCloseEvent>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QSettings>
#include <QShortcut>
#include <QToolButton>
#include <QVBoxLayout>

#include "app/view/canvas.hpp"
#include "app/view/inspector.hpp"
#include "app/view/nav_list.hpp"
#include "app/view/timeline.hpp"

namespace vl::app {

MainWindow::MainWindow(const Document& doc, QWidget* parent) : QMainWindow(parent), doc_(doc) {
    setWindowTitle(tr("Vision Lab"));
    auto* root = new QWidget;
    root->setObjectName(QStringLiteral("root"));
    auto* h = new QHBoxLayout(root);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(0);

    // ---------------------------------------------------------- navigation
    auto* left = new QWidget;
    left->setObjectName(QStringLiteral("navPanel"));
    left->setFixedWidth(232);
    auto* lv = new QVBoxLayout(left);
    lv->setContentsMargins(12, 18, 12, 14);
    lv->setSpacing(2);
    auto* brand = new QLabel(tr("Vision Lab"));
    brand->setObjectName(QStringLiteral("brand"));
    auto* tagline = new QLabel(tr("3D geometry and motion, from the maths up."));
    tagline->setObjectName(QStringLiteral("tagline"));
    tagline->setWordWrap(true);
    lv->addWidget(brand);
    lv->addWidget(tagline);
    lv->addSpacing(10);
    nav_ = new NavList;
    lv->addWidget(nav_, 1);
    auto* keys = new QLabel(tr("↑ ↓ topics · 1–9 views · F fit\nSpace play · ← → frames"));
    keys->setObjectName(QStringLiteral("keys"));
    lv->addWidget(keys);
    nav_->populate(doc);

    // ---------------------------------------------------------- canvas column
    auto* centre = new QWidget;
    centre->setObjectName(QStringLiteral("centre"));
    auto* cv = new QVBoxLayout(centre);
    cv->setContentsMargins(0, 0, 0, 0);
    cv->setSpacing(0);

    // Header, two rows: the title and the picture tools, then the view tabs and the run status.
    // Tabs get the full width of the canvas, so many views never widen the window.
    auto* header = new QWidget;
    header->setObjectName(QStringLiteral("header"));
    auto* hv = new QVBoxLayout(header);
    hv->setContentsMargins(22, 14, 16, 10);
    hv->setSpacing(10);
    auto* top = new QHBoxLayout;
    top->setSpacing(8);
    auto* titles = new QVBoxLayout;
    titles->setSpacing(1);
    crumb_ = new QLabel;
    crumb_->setObjectName(QStringLiteral("crumb"));
    title_ = new QLabel;
    title_->setObjectName(QStringLiteral("title"));
    title_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);  // a long title never widens the window
    title_->setMinimumWidth(160);
    crumb_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    titles->addWidget(crumb_);
    titles->addWidget(title_);
    top->addLayout(titles, 1);
    // The number that judges the result stays next to the picture, even when the inspector scrolls.
    headline_ = new QLabel;
    headline_->setObjectName(QStringLiteral("headline"));
    headline_->setTextFormat(Qt::RichText);
    headline_->setMaximumWidth(360);
    headline_->setMinimumWidth(1);  // a long progress line clips instead of widening the window
    top->addWidget(headline_);
    top->addSpacing(10);
    zoom_ = new QLabel;
    zoom_->setObjectName(QStringLiteral("zoom"));
    zoom_->setMinimumWidth(44);
    zoom_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    auto tool = [&](const QString& text, const QString& tip) {
        auto* b = new QToolButton;
        b->setText(text);
        b->setToolTip(tip);
        b->setObjectName(QStringLiteral("ghost"));
        b->setCursor(Qt::PointingHandCursor);
        return b;
    };
    fit_ = tool(tr("Fit"), tr("Fit the picture (F, or double-click)"));
    one_ = tool(tr("1:1"), tr("One image pixel per screen pixel"));
    export_ = tool(tr("Export"), tr("Save the picture with its overlays as PNG (Ctrl+E)"));
    code_ = tool(tr("Read the code"), tr("Our from-scratch implementation of this topic (Ctrl+K)"));
    top->addWidget(zoom_);
    top->addWidget(fit_);
    top->addWidget(one_);
    top->addWidget(export_);
    top->addWidget(code_);
    hv->addLayout(top);

    auto* second = new QHBoxLayout;
    second->setSpacing(12);
    tabs_ = new QWidget;
    tabs_->setObjectName(QStringLiteral("segmented"));
    tabsLayout_ = new QHBoxLayout(tabs_);
    tabsLayout_->setContentsMargins(0, 0, 0, 0);
    tabsLayout_->setSpacing(0);
    tabGroup_ = new QButtonGroup(this);
    // Many views (a topic compared with OpenCV has a dozen) scroll sideways instead of widening the window.
    auto* tabScroll = new QScrollArea;
    tabScroll->setObjectName(QStringLiteral("tabScroll"));
    tabScroll->setWidget(tabs_);
    tabScroll->setWidgetResizable(false);
    tabScroll->setFrameShape(QFrame::NoFrame);
    tabScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    tabScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    tabScroll->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    tabScroll->setFixedHeight(40);
    tabScroll_ = tabScroll;
    second->addWidget(tabScroll, 1);
    busyLabel_ = new QLabel;
    busyLabel_->setObjectName(QStringLiteral("busy"));
    busyLabel_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);  // long messages never push the layout
    busyLabel_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    busyLabel_->setMaximumWidth(260);
    second->addWidget(busyLabel_);
    // Our implementation is the topic; OpenCV's is the baseline it is measured against.
    compare_ = new QPushButton(tr("Compare with OpenCV"));
    compare_->setObjectName(QStringLiteral("compare"));
    compare_->setCheckable(true);
    compare_->setChecked(doc.compare());
    compare_->setCursor(Qt::PointingHandCursor);
    second->addWidget(compare_);

    hv->addLayout(second);
    cv->addWidget(header);

    progress_ = new QProgressBar;
    progress_->setObjectName(QStringLiteral("progress"));
    progress_->setTextVisible(false);
    progress_->setFixedHeight(2);
    progress_->setRange(0, 1000);
    progress_->setValue(0);
    cv->addWidget(progress_);

    canvas_ = new Canvas;
    cv->addWidget(canvas_, 1);

    auto* footer = new QWidget;
    footer->setObjectName(QStringLiteral("footer"));
    auto* fv = new QVBoxLayout(footer);
    fv->setContentsMargins(22, 8, 16, 10);
    fv->setSpacing(6);
    timeline_ = new Timeline;
    fv->addWidget(timeline_);
    status_ = new QLabel;
    status_->setObjectName(QStringLiteral("status"));
    status_->setText(QStringLiteral(" "));
    fv->addWidget(status_);
    cv->addWidget(footer);

    // ---------------------------------------------------------- inspector
    inspector_ = new Inspector;
    inspector_->setFixedWidth(340);

    h->addWidget(left);
    h->addWidget(centre, 1);
    h->addWidget(inspector_);
    setCentralWidget(root);

    connect(fit_, &QToolButton::clicked, canvas_, &Canvas::fit);
    connect(one_, &QToolButton::clicked, canvas_, [this] { canvas_->zoomTo(1.0); });
    connect(export_, &QToolButton::clicked, this, &MainWindow::exportRequested);
    connect(code_, &QToolButton::clicked, this, &MainWindow::codeRequested);
    connect(compare_, &QPushButton::toggled, this, &MainWindow::compareToggled);
    auto* codeKey = new QShortcut(QKeySequence(tr("Ctrl+K")), this);
    connect(codeKey, &QShortcut::activated, this, &MainWindow::codeRequested);
    connect(canvas_, &Canvas::zoomChanged, this, [this](double z) { zoom_->setText(QStringLiteral("%1%").arg(qRound(z * 100))); });
    connect(canvas_, &Canvas::hover, this, [this](const QString& t) { status_->setText(t.isEmpty() ? QStringLiteral(" ") : t); });
    connect(tabGroup_, &QButtonGroup::idClicked, this, &MainWindow::viewChosen);

    auto* openKey = new QShortcut(QKeySequence::Open, this);
    connect(openKey, &QShortcut::activated, this, &MainWindow::openRequested);
    auto* exportKey = new QShortcut(QKeySequence(tr("Ctrl+E")), this);
    connect(exportKey, &QShortcut::activated, this, &MainWindow::exportRequested);

    QSettings settings;
    if (!restoreGeometry(settings.value(QStringLiteral("window/geometry")).toByteArray())) resize(1440, 900);
}

void MainWindow::showModule() {
    const auto& s = doc_.current();
    crumb_->setText(s.info.group.toUpper());
    title_->setText(s.info.title);
    compare_->setVisible(!s.info.baseline.isEmpty());
    compare_->setToolTip(tr("Also run OpenCV's %1 on the same input and compare accuracy and time").arg(s.info.baseline));
    {
        const QSignalBlocker block(compare_);
        compare_->setChecked(doc_.compare());
    }
    nav_->select(doc_.currentIndex());
    inspector_->setModule(s);
    inspector_->setBoxToolChecked(false);
    canvas_->setBoxTool(false);
    rebuildViewTabs();
}

void MainWindow::rebuildViewTabs() {
    for (auto* b : tabGroup_->buttons()) {
        tabGroup_->removeButton(b);
        delete b;
    }
    const auto& s = doc_.current();
    if (!s.output || s.output->frames.empty()) {
        tabScroll_->hide();
        return;
    }
    const auto& views = s.output->frames.front().views;
    for (int i = 0; i < int(views.size()); ++i) {
        auto* b = new QPushButton(views[i].name);
        b->setCheckable(true);
        b->setCursor(Qt::PointingHandCursor);
        b->setToolTip(tr("Show %1 (%2)").arg(views[i].name).arg(i + 1));
        b->setProperty("segment", views.size() == 1 ? "only" : i == 0 ? "first" : i == int(views.size()) - 1 ? "last" : "middle");
        tabGroup_->addButton(b, i);
        tabsLayout_->addWidget(b);
        b->show();  // the strip is already on screen: show now rather than on a deferred event
    }
    if (auto* b = tabGroup_->button(s.view)) b->setChecked(true);
    tabs_->adjustSize();
    tabs_->show();
    tabScroll_->setVisible(views.size() > 1);
    if (auto* b = tabGroup_->button(s.view)) static_cast<QScrollArea*>(tabScroll_)->ensureWidgetVisible(b);
}

void MainWindow::showOutput(bool keepZoom) {
    const auto& s = doc_.current();
    const bool video = s.info.input == InputKind::Video;
    // Tabs follow the result's views (they can differ between topics, or appear with the first result).
    const int tabCount = int(tabGroup_->buttons().size());
    const int viewCount = s.output && !s.output->frames.empty() ? int(s.output->frames.front().views.size()) : 0;
    if (tabCount != viewCount || (viewCount > 0 && tabGroup_->button(0)->text() != s.output->frames.front().views[0].name))
        rebuildViewTabs();
    else if (auto* b = tabGroup_->button(s.view)) {
        b->setChecked(true);
        static_cast<QScrollArea*>(tabScroll_)->ensureWidgetVisible(b);
    }

    timeline_->setVisible(video);
    if (!s.output || s.output->frames.empty()) {
        if (s.inputs && !s.inputs->images.empty()) {
            // Before the first result: show the input so the window is never empty.
            View preview;
            preview.name = tr("Input");
            preview.image = toQImage(s.inputs->images.front());
            canvas_->show(&preview, nullptr, keepZoom);
        } else {
            canvas_->show(nullptr, nullptr, keepZoom);
        }
        if (video && s.inputs) timeline_->setFrames(0, int(s.inputs->images.size()), s.inputs->fps);
        headline_->clear();
        headline_->hide();
        inspector_->setResult(s);
        return;
    }
    const auto& frames = s.output->frames;
    const auto& frame = frames[std::min<size_t>(size_t(s.frame), frames.size() - 1)];
    const int view = std::clamp(s.view, 0, int(frame.views.size()) - 1);
    QString headline;
    for (const auto& st : s.output->stats)
        if (st.emphasis) headline = QStringLiteral("<span style='color:#9aa4b2'>%1</span>&nbsp;&nbsp;<b>%2</b>")
                                        .arg(st.label.toHtmlEscaped(), st.value.toHtmlEscaped());
    headline_->setText(headline);
    headline_->setVisible(!headline.isEmpty());  // no empty chip before the first result
    const std::vector<Shape>* extra = nullptr;
    for (const auto& cs : s.clickShapes)
        if (cs.view == view) extra = &cs.shapes;
    canvas_->show(&frame.views[size_t(view)], extra, keepZoom);
    if (video) {
        const int total = s.inputs ? int(s.inputs->images.size()) : int(frames.size());
        timeline_->setFrames(int(frames.size()), total, s.inputs ? s.inputs->fps : 25.0);
        timeline_->setFrame(s.frame);
    }
    inspector_->setResult(s);
    canvas_->setMessage({});
}

void MainWindow::showError(const QString& message) {
    inspector_->setResult(doc_.current());
    if (!message.isEmpty() && !doc_.current().output) canvas_->setMessage(message);
}

void MainWindow::setBusy(bool busy) {
    if (!busy) {
        progress_->setValue(0);
        busyLabel_->clear();
    }
    progress_->setProperty("busy", busy);
}

void MainWindow::setProgress(double fraction, const QString& message) {
    progress_->setValue(int(fraction * 1000));
    busyLabel_->setText(message);
}

void MainWindow::keyPressEvent(QKeyEvent* e) {
    const auto& s = doc_.current();
    const bool video = s.info.input == InputKind::Video;
    switch (e->key()) {
    case Qt::Key_F:
        canvas_->fit();
        return;
    case Qt::Key_Space:
        if (video) timeline_->togglePlay();
        return;
    case Qt::Key_Left:
        if (video) timeline_->step(-1);
        return;
    case Qt::Key_Right:
        if (video) timeline_->step(+1);
        return;
    default:
        break;
    }
    if (e->key() >= Qt::Key_1 && e->key() <= Qt::Key_9 && e->modifiers() == Qt::NoModifier) {
        const int v = e->key() - Qt::Key_1;
        if (tabGroup_->button(v)) emit viewChosen(v);
        return;
    }
    QMainWindow::keyPressEvent(e);
}

void MainWindow::closeEvent(QCloseEvent* e) {
    QSettings settings;
    settings.setValue(QStringLiteral("window/geometry"), saveGeometry());
    QMainWindow::closeEvent(e);
}

}  // namespace vl::app
