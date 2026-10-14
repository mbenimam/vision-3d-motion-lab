#pragma once
// The model: every module, and for each one its inputs, parameter values and latest result.
// It knows nothing about widgets or threads; it stores state and announces changes.

#include <memory>
#include <vector>

#include <QObject>

#include "app/module.hpp"

namespace vl::app {

struct ModuleState {
    std::unique_ptr<Module> module;
    Module::Info info;
    std::vector<ParamSpec> specs;
    std::vector<Sample> samples;
    Params params;
    int sampleIndex = 0;
    InputsPtr inputs;
    OutputPtr output;
    double lastRunMs = 0;
    QString error;
    int frame = 0;
    int view = 0;
    std::vector<Module::ClickShapes> clickShapes;
};

class Document : public QObject {
    Q_OBJECT
public:
    explicit Document(std::vector<std::unique_ptr<Module>> modules, QObject* parent = nullptr);

    int count() const { return int(states_.size()); }
    const ModuleState& state(int i) const { return states_.at(i); }
    const ModuleState& current() const { return states_.at(current_); }
    int currentIndex() const { return current_; }
    int indexOf(const QString& id) const;

    void setCurrent(int index);
    void setParam(const QString& key, const QVariant& value);
    void resetParams();
    void setSampleIndex(int index);
    void setInputs(InputsPtr inputs);
    void setOutput(OutputPtr output, double ms);
    void setPartialOutput(OutputPtr output);
    void setError(const QString& message);
    void setFrame(int frame);
    void setView(int view);
    void setClickShapes(std::vector<Module::ClickShapes> shapes);

signals:
    void currentChanged(int index);
    void paramsChanged();           // a value changed: the result is stale
    void inputsChanged();           // new pictures: the canvas refits
    void outputChanged();           // new result (or a partial one)
    void errorChanged(const QString& message);
    void frameChanged(int frame);
    void viewChanged(int view);
    void clickShapesChanged();

private:
    ModuleState& cur() { return states_.at(current_); }
    void saveParams(const ModuleState& s) const;
    std::vector<ModuleState> states_;
    int current_ = 0;
};

}  // namespace vl::app
