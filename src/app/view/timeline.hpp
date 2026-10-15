#pragma once
// Play / pause and a frame scrubber for video topics. Hidden for single images.

#include <QTimer>
#include <QWidget>

class QLabel;
class QSlider;
class QToolButton;

namespace vl::app {

class Timeline : public QWidget {
    Q_OBJECT
public:
    explicit Timeline(QWidget* parent = nullptr);
    void setFrames(int available, int total, double fps);  // available < total while computing
    void setFrame(int frame);
    void togglePlay();
    void step(int delta);
    bool playing() const { return timer_.isActive(); }

signals:
    void frameChosen(int frame);

private:
    void updateLabel();
    QToolButton* play_;
    QSlider* slider_;
    QLabel* label_;
    QTimer timer_;
    int available_ = 0, total_ = 0;
};

}  // namespace vl::app
