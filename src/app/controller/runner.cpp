#include "app/controller/runner.hpp"

#include <QPointer>
#include <QThreadPool>
#include <QtConcurrent>

namespace vl::app {

Runner::Runner(QObject* parent) : QObject(parent) {
    debounce_.setSingleShot(true);
    connect(&debounce_, &QTimer::timeout, this, &Runner::launch);
}

Runner::~Runner() {
    stop_.request_stop();
    QThreadPool::globalInstance()->waitForDone();
}

void Runner::submit(const Module* module, InputsPtr inputs, Params params, bool compare, int delayMs) {
    module_ = module;
    compare_ = compare;
    inputs_ = std::move(inputs);
    params_ = std::move(params);
    stop_.request_stop();  // the running job (if any) is now stale: let it stop early
    ++generation_;
    debounce_.start(delayMs);
}

void Runner::cancel() {
    debounce_.stop();
    stop_.request_stop();
    ++generation_;
}

void Runner::launch() {
    if (!module_ || !inputs_) return;
    stop_ = std::stop_source{};
    const auto gen = generation_;
    const auto stop = stop_.get_token();
    const Module* module = module_;
    const InputsPtr inputs = inputs_;
    const Params params = params_;
    QPointer<Runner> self(this);
    ++running_;
    emit started();

    // Callbacks run on the worker; they post to the UI thread, at most every 60 ms (progress)
    // and 150 ms (partial pictures), and only while this job is still the current one.
    auto clock = std::make_shared<QElapsedTimer>();
    clock->start();
    auto lastProgress = std::make_shared<std::atomic<qint64>>(-1000);
    auto lastPartial = std::make_shared<std::atomic<qint64>>(-1000);

    RunContext ctx;
    ctx.stop = stop;
    ctx.compare = compare_;
    ctx.progress = [=](double f, const std::string& msg) {
        const qint64 now = clock->elapsed();
        if (now - lastProgress->load() < 60 && f < 1.0) return;
        lastProgress->store(now);
        QMetaObject::invokeMethod(self, [=] {
            if (self && gen == self->generation_) emit self->progress(f, QString::fromStdString(msg));
        }, Qt::QueuedConnection);
    };
    ctx.partial = [=](OutputPtr out) {
        const qint64 now = clock->elapsed();
        if (now - lastPartial->load() < 150) return;
        lastPartial->store(now);
        QMetaObject::invokeMethod(self, [=] {
            if (self && gen == self->generation_) emit self->partial(out);
        }, Qt::QueuedConnection);
    };

    (void)QtConcurrent::run([=] {
        OutputPtr out;
        QString error;
        bool cancelled = false;
        vl::Stopwatch sw;
        try {
            out = std::make_shared<const Output>(module->compute(*inputs, params, ctx));
        } catch (const vl::Cancelled&) {
            cancelled = true;
        } catch (const std::exception& e) {
            error = QString::fromUtf8(e.what());
        }
        const double ms = sw.ms();
        QMetaObject::invokeMethod(self, [=] {
            if (!self) return;
            if (!cancelled && gen == self->generation_) {
                if (error.isEmpty())
                    emit self->finished(out, ms);
                else
                    emit self->failed(error);
            }
            self->jobDone();
        }, Qt::QueuedConnection);
    });
}

void Runner::jobDone() {
    if (--running_ == 0 && !debounce_.isActive()) emit idle();
}

void Runner::load(std::function<Inputs()> loader) {
    const auto gen = ++loadGeneration_;
    QPointer<Runner> self(this);
    (void)QtConcurrent::run([=] {
        InputsPtr in;
        QString error;
        try {
            in = std::make_shared<const Inputs>(loader());
            if (in->images.empty()) error = QStringLiteral("no images");
        } catch (const std::exception& e) {
            error = QString::fromUtf8(e.what());
        }
        QMetaObject::invokeMethod(self, [=] {
            if (!self || gen != self->loadGeneration_) return;
            if (error.isEmpty())
                emit self->loaded(in);
            else
                emit self->loadFailed(error);
        }, Qt::QueuedConnection);
    });
}

}  // namespace vl::app
