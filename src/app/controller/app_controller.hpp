#pragma once
// The controller: turns view gestures into model changes and computation requests, and model
// changes into view updates. Owns the Document (model), the Runner (worker) and the MainWindow (view).

#include <memory>

#include <QObject>

#include "app/controller/runner.hpp"
#include "app/model/document.hpp"

namespace vl::app {

class MainWindow;

class AppController : public QObject {
    Q_OBJECT
public:
    explicit AppController(QObject* parent = nullptr);
    ~AppController() override;

    void start(const QString& moduleId = {});
    MainWindow* window() const { return window_.get(); }
    Document& document() { return *doc_; }

    // Automation, used by --screenshot and by the README pictures.
    void selectSample(int index);
    void setParam(const QString& key, const QVariant& value);
    void setView(int view);
    void setFrame(int frame);
    void click(QPointF pixel);
    void whenIdle(std::function<void()> fn);  // runs once, after the next completed result
    void recompute(int delayMs);
    void syncInspector();  // show the model's parameter values (after changes made outside the panel)

private:
    void chooseModule(int index);
    void loadSample();
    void openFiles();
    void exportPicture();
    void refreshClickShapes();

    std::unique_ptr<Document> doc_;
    Runner runner_;
    std::unique_ptr<MainWindow> window_;
    std::function<void()> onIdle_;
    bool loading_ = false;
    QPointF lastClick_{-1, -1};
};

}  // namespace vl::app
