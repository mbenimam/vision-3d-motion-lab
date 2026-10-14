#include "app/model/document.hpp"

#include <algorithm>

#include <QSettings>

namespace vl::app {

Document::Document(std::vector<std::unique_ptr<Module>> modules, QObject* parent) : QObject(parent) {
    QSettings settings;
    compare_ = settings.value(QStringLiteral("compare"), false).toBool();
    for (auto& m : modules) {
        ModuleState s;
        s.info = m->info();
        s.specs = m->params();
        s.samples = m->samples();
        s.module = std::move(m);
        // Defaults, then whatever the user left last time (clamped, in case the range changed).
        settings.beginGroup(QStringLiteral("params/") + s.info.id);
        for (const auto& spec : s.specs) {
            double v = settings.value(spec.key, spec.value).toDouble();
            v = std::clamp(v, spec.min, spec.kind == ParamSpec::Kind::Toggle ? 1.0 : spec.max);
            if (spec.kind == ParamSpec::Kind::Real)
                s.params[spec.key] = v;
            else if (spec.kind == ParamSpec::Kind::Toggle)
                s.params[spec.key] = v != 0;
            else
                s.params[spec.key] = int(std::lround(v));
        }
        settings.endGroup();
        states_.push_back(std::move(s));
    }
}

int Document::indexOf(const QString& id) const {
    for (int i = 0; i < count(); ++i)
        if (states_[i].info.id == id) return i;
    return -1;
}

void Document::setCurrent(int index) {
    if (index < 0 || index >= count() || index == current_) return;
    current_ = index;
    emit currentChanged(index);
}

void Document::setParam(const QString& key, const QVariant& value) {
    auto& s = cur();
    if (s.params.value(key) == value) return;
    s.params[key] = value;
    if (key != QLatin1String("roi")) saveParams(s);
    emit paramsChanged();
}

void Document::resetParams() {
    auto& s = cur();
    const QVariant roi = s.params.value(QStringLiteral("roi"));
    s.params.clear();
    for (const auto& spec : s.specs) {
        if (spec.kind == ParamSpec::Kind::Real)
            s.params[spec.key] = spec.value;
        else if (spec.kind == ParamSpec::Kind::Toggle)
            s.params[spec.key] = spec.value != 0;
        else
            s.params[spec.key] = int(spec.value);
    }
    if (roi.isValid()) s.params[QStringLiteral("roi")] = roi;
    saveParams(s);
    emit paramsChanged();
}

void Document::saveParams(const ModuleState& s) const {
    QSettings settings;
    settings.beginGroup(QStringLiteral("params/") + s.info.id);
    for (auto it = s.params.cbegin(); it != s.params.cend(); ++it)
        if (it.key() != QLatin1String("roi")) settings.setValue(it.key(), it.value().toDouble());
    settings.endGroup();
}

void Document::setSampleIndex(int index) { cur().sampleIndex = index; }

void Document::setInputs(InputsPtr inputs) {
    auto& s = cur();
    s.inputs = std::move(inputs);
    s.output.reset();
    s.error.clear();
    s.frame = 0;
    s.clickShapes.clear();
    s.params.remove(QStringLiteral("roi"));  // a box drawn on other pictures means nothing here
    emit inputsChanged();
    emit outputChanged();
}

void Document::setOutput(OutputPtr output, double ms) {
    auto& s = cur();
    s.output = std::move(output);
    s.lastRunMs = ms;
    s.error.clear();
    if (s.output && !s.output->frames.empty()) {
        const auto& views = s.output->frames.front().views;
        // Open on the topic's result, or on "result · ours | OpenCV" when comparing.
        if (!s.viewChosen && !s.info.defaultView.isEmpty()) {
            const QString pair = s.info.defaultView + QStringLiteral(" · ours | OpenCV");
            for (int i = 0; i < int(views.size()); ++i)
                if (views[size_t(i)].name == s.info.defaultView) s.view = i;
            for (int i = 0; i < int(views.size()); ++i)
                if (views[size_t(i)].name == pair) s.view = i;
        }
        s.frame = std::min(s.frame, int(s.output->frames.size()) - 1);
        s.view = std::min(s.view, int(s.output->frames.front().views.size()) - 1);
    }
    // Clicked overlays (an epipolar line) belong to the previous result: recompute them from the new one.
    s.clickShapes.clear();
    emit outputChanged();
    emit errorChanged({});
}

void Document::setPartialOutput(OutputPtr output) {
    auto& s = cur();
    s.output = std::move(output);
    emit outputChanged();
}

void Document::setError(const QString& message) {
    cur().error = message;
    emit errorChanged(message);
}

void Document::setFrame(int frame) {
    auto& s = cur();
    const int n = s.output ? int(s.output->frames.size()) : 1;
    frame = std::clamp(frame, 0, std::max(0, n - 1));
    if (frame == s.frame) return;
    s.frame = frame;
    emit frameChanged(frame);
}

void Document::setView(int view) {
    auto& s = cur();
    const int n = s.output && !s.output->frames.empty() ? int(s.output->frames.front().views.size()) : 1;
    view = std::clamp(view, 0, std::max(0, n - 1));
    s.viewChosen = true;
    if (view == s.view) return;
    s.view = view;
    emit viewChanged(view);
}

void Document::setCompare(bool on) {
    if (on == compare_) return;
    compare_ = on;
    QSettings().setValue(QStringLiteral("compare"), on);
    for (auto& s : states_) s.output.reset();  // every result is stale now
    emit paramsChanged();
}

void Document::setClickShapes(std::vector<Module::ClickShapes> shapes) {
    cur().clickShapes = std::move(shapes);
    emit clickShapesChanged();
}

}  // namespace vl::app
