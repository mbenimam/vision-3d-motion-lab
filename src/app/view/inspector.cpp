#include "app/view/inspector.hpp"

#include <algorithm>
#include <cmath>

#include <array>

#include <QButtonGroup>
#include <QStyle>
#include <QCheckBox>
#include <QComboBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPainter>
#include <QPainterPath>
#include <QPushButton>
#include <QScrollBar>
#include <QSlider>
#include <QToolButton>
#include <QVBoxLayout>

namespace vl::app {

// ------------------------------------------------------------------ one parameter
// Label on the left, value on the right, control underneath. Choices with up to four options
// are a segmented control (all options visible, one click); longer lists are a combo box.
class ParamRow : public QWidget {
    Q_OBJECT
public:
    explicit ParamRow(const ParamSpec& spec, QWidget* parent = nullptr) : QWidget(parent), spec_(spec) {
        auto* v = new QVBoxLayout(this);
        v->setContentsMargins(0, 0, 0, 0);
        v->setSpacing(4);
        setToolTip(spec.help);
        if (spec.kind == ParamSpec::Kind::Toggle) {
            check_ = new QCheckBox(spec.label);
            check_->setObjectName(QStringLiteral("toggle"));
            check_->setCursor(Qt::PointingHandCursor);
            v->addWidget(check_);
            connect(check_, &QCheckBox::toggled, this, [this](bool on) {
                if (!quiet_) emit changed(spec_.key, on);
            });
            return;
        }
        auto* head = new QHBoxLayout;
        auto* label = new QLabel(spec.label);
        label->setObjectName(QStringLiteral("paramLabel"));
        label->setWordWrap(true);
        head->addWidget(label, 1);
        value_ = new QLabel(this);
        value_->setObjectName(QStringLiteral("paramValue"));
        value_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

        switch (spec.kind) {
        case ParamSpec::Kind::Int:
        case ParamSpec::Kind::Real: {
            head->addWidget(value_);
            v->addLayout(head);
            slider_ = new QSlider(Qt::Horizontal);
            slider_->setRange(0, steps());
            slider_->setPageStep(std::max(1, steps() / 10));
            v->addWidget(slider_);
            connect(slider_, &QSlider::valueChanged, this, [this](int i) {
                showValue(fromIndex(i));
                if (!quiet_) emit changed(spec_.key, current());
            });
            break;
        }
        case ParamSpec::Kind::Choice: {
            // All options visible in one row when they fit the panel; otherwise a combo box.
            if (spec.choices.size() <= 4 && spec.choices.join(QString()).size() <= 30) {
                v->addLayout(head);
                auto* seg = new QWidget;
                seg->setObjectName(QStringLiteral("segmented"));
                auto* h = new QHBoxLayout(seg);
                h->setContentsMargins(0, 0, 0, 0);
                h->setSpacing(0);
                group_ = new QButtonGroup(this);
                for (int i = 0; i < spec.choices.size(); ++i) {
                    auto* b = new QPushButton(spec.choices[i]);
                    b->setCheckable(true);
                    b->setProperty("segment", i == 0 ? "first" : i == spec.choices.size() - 1 ? "last" : "middle");
                    b->setCursor(Qt::PointingHandCursor);
                    group_->addButton(b, i);
                    h->addWidget(b, 1);
                }
                v->addWidget(seg);
                connect(group_, &QButtonGroup::idClicked, this, [this](int id) {
                    if (!quiet_) emit changed(spec_.key, id);
                });
            } else {
                v->addLayout(head);
                combo_ = new QComboBox;
                combo_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
                combo_->setMinimumContentsLength(12);
                combo_->addItems(spec.choices);
                v->addWidget(combo_);
                connect(combo_, &QComboBox::currentIndexChanged, this, [this](int i) {
                    if (!quiet_) emit changed(spec_.key, i);
                });
            }
            break;
        }
        case ParamSpec::Kind::Toggle:
            break;
        }
    }

    void setValue(const QVariant& value) {
        quiet_ = true;
        switch (spec_.kind) {
        case ParamSpec::Kind::Int:
        case ParamSpec::Kind::Real:
            slider_->setValue(toIndex(value.toDouble()));
            showValue(fromIndex(slider_->value()));
            break;
        case ParamSpec::Kind::Choice:
            if (group_) {
                if (auto* b = group_->button(value.toInt())) b->setChecked(true);
            } else {
                combo_->setCurrentIndex(value.toInt());
            }
            break;
        case ParamSpec::Kind::Toggle:
            check_->setChecked(value.toBool());
            break;
        }
        quiet_ = false;
    }

signals:
    void changed(const QString& key, const QVariant& value);

private:
    int steps() const { return std::max(1, int(std::lround((spec_.max - spec_.min) / spec_.step))); }
    double fromIndex(int i) const { return std::min(spec_.max, spec_.min + i * spec_.step); }
    int toIndex(double v) const { return int(std::lround((v - spec_.min) / spec_.step)); }
    QVariant current() const {
        const double v = fromIndex(slider_->value());
        return spec_.kind == ParamSpec::Kind::Int ? QVariant(int(std::lround(v))) : QVariant(v);
    }
    void showValue(double v) {
        value_->setText(QString::number(v, 'f', spec_.kind == ParamSpec::Kind::Int ? 0 : spec_.decimals) + spec_.suffix);
    }

    ParamSpec spec_;
    QLabel* value_ = nullptr;
    QSlider* slider_ = nullptr;
    QButtonGroup* group_ = nullptr;
    QComboBox* combo_ = nullptr;
    QCheckBox* check_ = nullptr;
    bool quiet_ = false;
};

// ------------------------------------------------------------------ a tiny line chart
class Sparkline : public QWidget {
public:
    Sparkline(const Series& s, QWidget* parent = nullptr) : QWidget(parent), s_(s) { setFixedHeight(54); }
    const QString& label() const { return s_.label; }
    void setSeries(const Series& s) {
        s_ = s;
        update();  // repaint in place: no widget is created or destroyed while a video plays
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        const QRectF r = rect().adjusted(1, 18, -1, -4);
        p.setPen(colors::muted);
        QFont f = font();
        f.setPointSizeF(f.pointSizeF() * 0.9);
        p.setFont(f);
        p.drawText(rect().adjusted(0, 0, 0, 0), Qt::AlignLeft | Qt::AlignTop, s_.label);
        if (s_.values.empty()) return;
        const auto [lo, hi] = std::minmax_element(s_.values.begin(), s_.values.end());
        p.setPen(QColor(0xe6, 0xe6, 0xe6));
        p.drawText(rect(), Qt::AlignRight | Qt::AlignTop, QString::number(s_.values.back(), 'g', 4));
        const double span = (*hi - *lo) > 1e-12 ? (*hi - *lo) : 1.0;
        const auto n = s_.values.size();
        QPainterPath path;
        for (size_t i = 0; i < n; ++i) {
            const double x = r.left() + (n == 1 ? r.width() : r.width() * double(i) / double(n - 1));
            const double y = r.bottom() - r.height() * (s_.values[i] - *lo) / span;
            i == 0 ? path.moveTo(x, y) : path.lineTo(x, y);
        }
        QPen pen(colors::accent, 1.6);
        p.setPen(pen);
        p.drawPath(path);
        p.setBrush(colors::accent);
        p.setPen(Qt::NoPen);
        p.drawEllipse(path.currentPosition(), 2.5, 2.5);
    }

private:
    Series s_;
};

// ------------------------------------------------------------------ the panel
Inspector::Inspector(QWidget* parent) : QScrollArea(parent) {
    setObjectName(QStringLiteral("inspector"));
    setWidgetResizable(true);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setFrameShape(NoFrame);
    auto* body = new QWidget;
    body->setObjectName(QStringLiteral("inspectorBody"));
    body->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    layout_ = new QVBoxLayout(body);
    layout_->setContentsMargins(20, 18, 18, 24);
    layout_->setSpacing(0);
    setWidget(body);

    // How it works
    auto* how = section(tr("How it works"));
    auto* howL = static_cast<QVBoxLayout*>(how->layout());
    idea_ = new QLabel;
    idea_->setObjectName(QStringLiteral("idea"));
    idea_->setWordWrap(true);
    idea_->setTextFormat(Qt::RichText);
    try_ = new QLabel;
    try_->setObjectName(QStringLiteral("try"));
    try_->setWordWrap(true);
    try_->setTextFormat(Qt::RichText);
    howL->addWidget(idea_);
    howL->addWidget(try_);

    // Input
    open_ = new QToolButton;
    open_->setText(tr("Open…"));
    open_->setObjectName(QStringLiteral("linkButton"));
    open_->setCursor(Qt::PointingHandCursor);
    open_->setToolTip(tr("Open your own image(s) or video (Ctrl+O)"));
    auto* input = section(tr("Input"), open_);
    auto* inL = static_cast<QVBoxLayout*>(input->layout());
    samples_ = new QComboBox;
    samples_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);  // long names never widen the panel
    samples_->setMinimumContentsLength(12);
    inL->addWidget(samples_);
    boxTool_ = new QPushButton(tr("Draw the box to track"));
    boxTool_->setObjectName(QStringLiteral("secondary"));
    boxTool_->setCheckable(true);
    boxTool_->setCursor(Qt::PointingHandCursor);
    boxHint_ = new QLabel(tr("Drag a rectangle around the object on the picture."));
    boxHint_->setObjectName(QStringLiteral("hint"));
    boxHint_->setWordWrap(true);
    inL->addWidget(boxTool_);
    inL->addWidget(boxHint_);
    connect(samples_, &QComboBox::activated, this, &Inspector::sampleChosen);
    connect(open_, &QToolButton::clicked, this, &Inspector::openRequested);
    connect(boxTool_, &QPushButton::toggled, this, [this](bool on) {
        boxHint_->setVisible(on);
        emit boxToolToggled(on);
    });

    // Parameters
    auto* reset = new QToolButton;
    reset->setText(tr("Reset"));
    reset->setObjectName(QStringLiteral("linkButton"));
    reset->setCursor(Qt::PointingHandCursor);
    reset->setToolTip(tr("Back to the default values"));
    connect(reset, &QToolButton::clicked, this, &Inspector::resetRequested);
    paramsBox_ = section(tr("Parameters"), reset);
    paramsLayout_ = static_cast<QVBoxLayout*>(paramsBox_->layout());
    paramsLayout_->setSpacing(14);

    // Result
    statsBox_ = section(tr("Result"));
    auto* rl = static_cast<QVBoxLayout*>(statsBox_->layout());
    error_ = new QLabel;
    error_->setObjectName(QStringLiteral("error"));
    error_->setWordWrap(true);
    rl->addWidget(error_);
    // Ours vs OpenCV: one row per measurement, the better value marked.
    compareBox_ = new QWidget;
    compareBox_->setObjectName(QStringLiteral("compareTable"));
    compareGrid_ = new QGridLayout(compareBox_);
    compareGrid_->setContentsMargins(10, 8, 10, 8);
    compareGrid_->setHorizontalSpacing(10);
    compareGrid_->setVerticalSpacing(5);
    rl->addWidget(compareBox_);
    auto* grid = new QWidget;
    statsGrid_ = new QGridLayout(grid);
    statsGrid_->setContentsMargins(0, 0, 0, 0);
    statsGrid_->setHorizontalSpacing(12);
    statsGrid_->setVerticalSpacing(6);
    rl->addWidget(grid);
    seriesLayout_ = new QVBoxLayout;
    seriesLayout_->setSpacing(10);
    rl->addLayout(seriesLayout_);
    note_ = new QLabel;
    note_->setObjectName(QStringLiteral("hint"));
    note_->setWordWrap(true);
    rl->addWidget(note_);
    statsBox_->setProperty("last", true);
    layout_->addStretch(1);
}

QWidget* Inspector::section(const QString& title, QWidget* trailing) {
    auto* box = new QWidget;
    box->setObjectName(QStringLiteral("section"));
    auto* v = new QVBoxLayout(box);
    v->setContentsMargins(0, 0, 0, 18);
    v->setSpacing(10);
    auto* head = new QHBoxLayout;
    auto* t = new QLabel(title.toUpper());
    t->setObjectName(QStringLiteral("sectionTitle"));
    head->addWidget(t, 1);
    if (trailing) head->addWidget(trailing);
    v->addLayout(head);
    layout_->addWidget(box);
    auto* rule = new QWidget;
    rule->setObjectName(QStringLiteral("rule"));
    rule->setFixedHeight(1);
    layout_->addWidget(rule);
    layout_->addSpacing(16);
    return box;
}

void Inspector::setModule(const ModuleState& s) {
    idea_->setText(s.info.idea);
    try_->setText(QStringLiteral("<b>Try:</b> ") + s.info.tryHint);
    try_->setVisible(!s.info.tryHint.isEmpty());

    samples_->clear();
    for (const auto& sm : s.samples) samples_->addItem(sm.name);
    samples_->setCurrentIndex(s.sampleIndex);
    const bool roi = s.module->wantsRoi();
    boxTool_->setVisible(roi);
    boxHint_->setVisible(roi && boxTool_->isChecked());

    for (auto& [k, row] : rows_) delete row;
    rows_.clear();
    for (const auto& spec : s.specs) {
        auto* row = new ParamRow(spec);
        connect(row, &ParamRow::changed, this, &Inspector::paramChanged);
        paramsLayout_->addWidget(row);
        rows_[spec.key] = row;
    }
    paramsBox_->setVisible(!s.specs.empty());
    syncParams(s.params);
    setResult(s);
    verticalScrollBar()->setValue(0);
}

void Inspector::syncParams(const Params& p) {
    for (auto& [k, row] : rows_) row->setValue(p.value(k));
}

void Inspector::setSampleIndex(int index) {
    const QSignalBlocker block(samples_);
    samples_->setCurrentIndex(index);
}

void Inspector::setBoxToolChecked(bool on) {
    const QSignalBlocker block(boxTool_);
    boxTool_->setChecked(on);
    boxHint_->setVisible(on);
}

// Called for every new result and, while a video plays, for every frame. Rows are updated in place
// when the set of rows is unchanged (no widget churn, so nothing flickers); they are rebuilt only when
// the rows themselves change (another topic, comparison switched on or off).
void Inspector::setResult(const ModuleState& s) {
    error_->setText(s.error);
    error_->setVisible(!s.error.isEmpty());

    // Ours vs OpenCV.
    std::vector<std::array<QString, 4>> compareRows;  // metric, ours, OpenCV, which wins ("ours"/"opencv"/"")
    if (s.output)
        for (const auto& c : s.output->comparison) {
            const bool tie = std::abs(c.ours - c.opencv) < 0.5 * std::pow(10.0, -c.decimals);
            const bool oursWins = !tie && ((c.ours < c.opencv) == c.lowerIsBetter);
            compareRows.push_back({c.metric, fmt(c.ours, c.decimals) + c.unit, fmt(c.opencv, c.decimals) + c.unit,
                                   tie ? QString() : oursWins ? QStringLiteral("ours") : QStringLiteral("opencv")});
        }
    compareBox_->setVisible(!compareRows.empty());
    bool sameCompare = compareCells_.size() == compareRows.size();
    for (size_t i = 0; sameCompare && i < compareRows.size(); ++i) sameCompare = compareCells_[i][0]->text() == compareRows[i][0];
    if (!sameCompare) {
        while (auto* item = compareGrid_->takeAt(0)) {
            delete item->widget();
            delete item;
        }
        compareCells_.clear();
        auto cell = [this](const QString& text, const char* name, int r, int c, Qt::Alignment a) {
            auto* l = new QLabel(text);
            l->setObjectName(QString::fromLatin1(name));
            l->setWordWrap(c == 0);
            compareGrid_->addWidget(l, r, c, a);
            return l;
        };
        if (!compareRows.empty()) {
            cell({}, "compareHead", 0, 0, Qt::AlignLeft);
            cell(tr("Ours"), "compareHead", 0, 1, Qt::AlignRight);
            cell(tr("OpenCV"), "compareHead", 0, 2, Qt::AlignRight);
        }
        for (size_t i = 0; i < compareRows.size(); ++i) {
            const int r = int(i) + 1;
            compareCells_.push_back({cell(compareRows[i][0], "statLabel", r, 0, Qt::AlignLeft),
                                     cell({}, "statValue", r, 1, Qt::AlignRight), cell({}, "statValue", r, 2, Qt::AlignRight)});
        }
        compareGrid_->setColumnStretch(0, 1);
    }
    for (size_t i = 0; i < compareRows.size(); ++i) {
        for (int k = 1; k <= 2; ++k) {
            QLabel* l = compareCells_[i][size_t(k)];
            l->setText(compareRows[i][size_t(k)]);
            const bool wins = compareRows[i][3] == (k == 1 ? QStringLiteral("ours") : QStringLiteral("opencv"));
            const QString name = wins ? QStringLiteral("winner") : QStringLiteral("statValue");
            if (l->objectName() != name) {
                l->setObjectName(name);
                l->style()->unpolish(l);
                l->style()->polish(l);
            }
        }
    }

    // Numbers of the result, then of the current frame.
    std::vector<Stat> stats;
    if (s.output) {
        stats = s.output->stats;
        if (!s.output->frames.empty()) {
            const auto& fs = s.output->frames[std::min<size_t>(size_t(s.frame), s.output->frames.size() - 1)].stats;
            stats.insert(stats.end(), fs.begin(), fs.end());
        }
        if (s.lastRunMs > 0) stats.push_back({tr("Run time"), fmtMs(s.lastRunMs)});
    }
    bool sameStats = statRows_.size() == stats.size();
    for (size_t i = 0; sameStats && i < stats.size(); ++i)
        sameStats = statRows_[i].first->text() == stats[i].label &&
                    (statRows_[i].second->objectName() == QStringLiteral("statKey")) == stats[i].emphasis;
    if (!sameStats) {
        while (auto* item = statsGrid_->takeAt(0)) {
            delete item->widget();
            delete item;
        }
        statRows_.clear();
        for (size_t i = 0; i < stats.size(); ++i) {
            auto* l = new QLabel(stats[i].label);
            l->setObjectName(QStringLiteral("statLabel"));
            l->setWordWrap(true);
            auto* v = new QLabel;
            v->setObjectName(stats[i].emphasis ? QStringLiteral("statKey") : QStringLiteral("statValue"));
            v->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
            v->setTextInteractionFlags(Qt::TextSelectableByMouse);
            statsGrid_->addWidget(l, int(i), 0);
            statsGrid_->addWidget(v, int(i), 1);
            statRows_.emplace_back(l, v);
        }
    }
    for (size_t i = 0; i < stats.size(); ++i) statRows_[i].second->setText(stats[i].value);

    // Sparklines.
    const std::vector<Series> series = s.output ? s.output->series : std::vector<Series>{};
    bool sameSeries = sparks_.size() == series.size();
    for (size_t i = 0; sameSeries && i < series.size(); ++i) sameSeries = sparks_[i]->label() == series[i].label;
    if (!sameSeries) {
        while (auto* item = seriesLayout_->takeAt(0)) {
            delete item->widget();
            delete item;
        }
        sparks_.clear();
        for (const auto& se : series) {
            sparks_.push_back(new Sparkline(se));
            seriesLayout_->addWidget(sparks_.back());
        }
    } else {
        for (size_t i = 0; i < series.size(); ++i) sparks_[i]->setSeries(series[i]);
    }

    note_->setText(s.output ? s.output->note : QString());
    note_->setVisible(!note_->text().isEmpty());
    statsBox_->setVisible(s.output || !s.error.isEmpty());
}

}  // namespace vl::app

#include "inspector.moc"
