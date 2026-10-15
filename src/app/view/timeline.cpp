#include "app/view/timeline.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QSlider>
#include <QToolButton>

namespace vl::app {

Timeline::Timeline(QWidget* parent) : QWidget(parent) {
    setObjectName(QStringLiteral("timeline"));
    auto* h = new QHBoxLayout(this);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(12);
    play_ = new QToolButton;
    play_->setObjectName(QStringLiteral("play"));
    play_->setText(QStringLiteral("▶"));
    play_->setToolTip(tr("Play / pause (Space)"));
    play_->setCursor(Qt::PointingHandCursor);
    slider_ = new QSlider(Qt::Horizontal);
    slider_->setObjectName(QStringLiteral("scrubber"));
    label_ = new QLabel;
    label_->setObjectName(QStringLiteral("frameLabel"));
    label_->setMinimumWidth(84);
    label_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    h->addWidget(play_);
    h->addWidget(slider_, 1);
    h->addWidget(label_);

    connect(play_, &QToolButton::clicked, this, &Timeline::togglePlay);
    connect(slider_, &QSlider::valueChanged, this, [this](int v) {
        updateLabel();
        emit frameChosen(v);
    });
    connect(&timer_, &QTimer::timeout, this, [this] {
        if (available_ <= 1) return;
        slider_->setValue((slider_->value() + 1) % available_);
    });
}

void Timeline::setFrames(int available, int total, double fps) {
    available_ = available;
    total_ = total;
    const QSignalBlocker block(slider_);
    slider_->setRange(0, std::max(0, available - 1));
    timer_.setInterval(int(1000.0 / std::clamp(fps, 1.0, 60.0)));
    updateLabel();
}

void Timeline::setFrame(int frame) {
    const QSignalBlocker block(slider_);
    slider_->setValue(frame);
    updateLabel();
}

void Timeline::togglePlay() {
    if (timer_.isActive()) {
        timer_.stop();
        play_->setText(QStringLiteral("▶"));
    } else {
        timer_.start();
        play_->setText(QStringLiteral("❚❚"));
    }
}

void Timeline::step(int delta) {
    if (available_ > 0) slider_->setValue(std::clamp(slider_->value() + delta, 0, available_ - 1));
}

void Timeline::updateLabel() {
    QString text = QStringLiteral("%1 / %2").arg(slider_->value() + 1).arg(std::max(1, total_));
    label_->setText(text);
}

}  // namespace vl::app
