#pragma once
// What the three optical-flow topics show in common, so they read as one family:
//
//   0 Frame 1   frame 0 with the flow as arrows on a grid
//   1 Frame 2   frame 1 (click: where the pixel went)
//   2 <method>  the mechanism of the method (match cost, confidence, data weight), built by the module
//   3 Flow      the Middlebury colour code of the estimate
//   4 Colour key the colour wheel at the same scale (click: where the vector sits on it)
//   5 Warped    frame 2 warped back by the flow: should look like frame 1 where the flow is right
//   6 Truth     the ground-truth flow, same colour scale            (only with ground truth)
//   7 Error     endpoint error map                                  (only with ground truth)
//   last        OpenCV · Flow, the baseline's estimate at the same scale (only with Compare on)
//
// The Presenter precomputes what does not depend on the estimate, so partial results (one per few
// iterations) stay cheap.

#include <array>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "app/module.hpp"

namespace vl::app::flow_views {

enum ViewIndex : int { kFrame1, kFrame2, kMechanism, kFlow, kKey, kWarped, kTruth, kError };

// Extra numbers a module needs back in onClick (the API hands it only the Output): stored as a
// 1 x kMetaSize CV_32F row in the Colour key view's data. Slot 0 is the colour scale.
enum Meta : int { kScale, kExtra0, kExtra1, kExtra2, kMetaSize };
using MetaValues = std::array<float, 3>;  // kExtra0..kExtra2
float meta(const Output& out, Meta m);

class Presenter {
public:
    Presenter(const Inputs& in, int arrowStep);

    // Views and the error statistics (first, with one emphasis) for one estimate; the module appends
    // its own stats and series. The mechanism view is inserted third.
    Output present(const cv::Mat& flow, View mechanism, const MetaValues& extra = {}) const;

    bool hasTruth() const { return !truth_.empty(); }
    // The colour scale (px at the rim of the wheel) used for this estimate: the truth's when known.
    float colourScale(const cv::Mat& flow) const;
    double truthMax() const { return truthMax_; }  // largest known |w_gt| (px), 0 without truth

private:
    const Inputs& in_;
    int arrowStep_;
    cv::Mat truth_;
    double truthMax_ = 0;
    float truthScale_ = 0;  // robust colour scale of the truth (px)
    QImage frame1_, frame2_, truthImage_, keyImage_;
};

// One extra line of the ours-vs-OpenCV table, computed by the module (e.g. pixels with an estimate).
using ExtraComparison = std::optional<Comparison>;

// The OpenCV baseline next to ours (when RunContext::compare is set): its flow as the view
// "OpenCV · Flow" at the same colour scale, and the same measurements side by side in
// Output::comparison: endpoint and angular error, share of EPE > 1 px (with ground truth), the
// module's extra line, the brightness residual, and the run times.
void addComparison(Output& out, const Inputs& in, const Presenter& pres, const cv::Mat& ours, double oursMs,
                   const cv::Mat& theirs, double theirsMs, const QString& method, const ExtraComparison& extra = {});

// Share of pixels with a finite estimate, in %.
double coveragePercent(const cv::Mat& flow);

// The click shared by the topics: the pixel's vector (and the true one) on every view, where the
// pixel lands in frame 2, and its place on the colour wheel. Empty when the click is outside.
std::vector<Module::ClickShapes> clickShapes(const Output& out, QPointF pixel);

// Where to anchor a click label next to point p so it stays on a picture of this size.
QPointF labelAt(QPointF p, cv::Size size);

// Turbo colour of a value in [lo, hi] (for overlays drawn as shapes).
QColor turbo(double v, double lo, double hi);

}  // namespace vl::app::flow_views
