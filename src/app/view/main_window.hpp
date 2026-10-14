#pragma once
// The window: navigation | canvas (header, picture, timeline, status) | inspector.
// A passive view: it renders the Document and turns user gestures into signals.

#include <QMainWindow>

#include "app/model/document.hpp"

class QButtonGroup;
class QLabel;
class QProgressBar;
class QToolButton;
class QHBoxLayout;

namespace vl::app {

class Canvas;
class Inspector;
class NavList;
class Timeline;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(const Document& doc, QWidget* parent = nullptr);

    Canvas* canvas() const { return canvas_; }
    Inspector* inspector() const { return inspector_; }
    NavList* nav() const { return nav_; }
    Timeline* timeline() const { return timeline_; }

    void showModule();          // a new topic: header, view tabs, inspector
    void showOutput(bool keepZoom);  // the current frame / view of the current result
    void showError(const QString& message);
    void setBusy(bool busy);
    void setProgress(double fraction, const QString& message);

signals:
    void viewChosen(int view);
    void exportRequested();
    void openRequested();

protected:
    void keyPressEvent(QKeyEvent* e) override;
    void closeEvent(QCloseEvent* e) override;

private:
    void rebuildViewTabs();

    const Document& doc_;
    NavList* nav_;
    Canvas* canvas_;
    Inspector* inspector_;
    Timeline* timeline_;
    QLabel* crumb_;
    QLabel* title_;
    QWidget* tabs_;
    QHBoxLayout* tabsLayout_;
    QButtonGroup* tabGroup_;
    QToolButton* fit_;
    QToolButton* one_;
    QToolButton* export_;
    QLabel* zoom_;
    QProgressBar* progress_;
    QLabel* status_;
    QLabel* busyLabel_;
};

}  // namespace vl::app
