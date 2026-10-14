#pragma once
// Runs one module computation at a time on a worker thread.
//
// - submit() debounces: a burst of slider moves becomes one run.
// - A new submit cancels the running job through its std::stop_token; the old result is dropped.
// - Progress and partial results are forwarded to the UI thread, throttled so painting never lags.

#include <atomic>
#include <memory>
#include <stop_token>

#include <QElapsedTimer>
#include <QObject>
#include <QTimer>

#include "app/module.hpp"

namespace vl::app {

class Runner : public QObject {
    Q_OBJECT
public:
    explicit Runner(QObject* parent = nullptr);
    ~Runner() override;

    // Compute module(inputs, params) after delayMs of quiet. Replaces any pending or running request.
    void submit(const Module* module, InputsPtr inputs, Params params, int delayMs);
    // Load a sample (or user files) off the UI thread.
    void load(std::function<Inputs()> loader);
    void cancel();
    bool busy() const { return running_ > 0; }

signals:
    void started();
    void progress(double fraction, const QString& message);
    void partial(vl::app::OutputPtr output);
    void finished(vl::app::OutputPtr output, double ms);
    void failed(const QString& message);
    void loaded(vl::app::InputsPtr inputs);
    void loadFailed(const QString& message);
    void idle();

private:
    void launch();
    void jobDone();

    QTimer debounce_;
    const Module* module_ = nullptr;
    InputsPtr inputs_;
    Params params_;
    std::stop_source stop_;
    std::uint64_t generation_ = 0;      // a result is shown only if its generation is still current
    std::uint64_t loadGeneration_ = 0;
    int running_ = 0;
};

}  // namespace vl::app
