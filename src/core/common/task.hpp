#pragma once
// Cooperative cancellation and progress reporting for long computations.
// The core never depends on Qt: the app passes a std::stop_token and a plain callback.

#include <chrono>
#include <exception>
#include <functional>
#include <stop_token>
#include <string>

namespace vl {

// Thrown by checkStop() when the caller asked to stop; the app catches it and drops the job.
struct Cancelled : std::exception {
    const char* what() const noexcept override { return "cancelled"; }
};

inline void checkStop(const std::stop_token& st) {
    if (st.stop_requested()) throw Cancelled{};
}

// fraction in [0, 1] and a short human message ("iteration 3 / 5").
using ProgressFn = std::function<void(double fraction, const std::string& message)>;

inline void report(const ProgressFn& fn, double fraction, const std::string& message) {
    if (fn) fn(fraction, message);
}

// Wall-clock milliseconds since construction.
class Stopwatch {
public:
    Stopwatch() : t0_(std::chrono::steady_clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
    }

private:
    std::chrono::steady_clock::time_point t0_;
};

}  // namespace vl
