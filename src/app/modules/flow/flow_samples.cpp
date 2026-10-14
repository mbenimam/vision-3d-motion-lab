#include "app/modules/flow/flow_samples.hpp"

#include "core/common/synthetic.hpp"
#include "core/flow/flow_eval.hpp"
#include "core/flow/flow_scenes.hpp"

namespace vl::app::flow_samples {
namespace {

Inputs fromPair(const QString& name, const vl::synth::FlowPair& p) {
    Inputs in;
    in.name = name;
    in.images = {p.frame0, p.frame1};
    in.truth = {p.flow};
    return in;
}

}  // namespace

std::vector<Sample> all() {
    std::vector<Sample> out;
    // Rotation by 2° and 2 % zoom about the centre plus a shift: up to 11 px of motion in the corners,
    // more than one scale can linearise, so the pyramid matters.
    out.push_back({QStringLiteral("Synthetic: rotation and zoom"), [] {
                       return fromPair(QStringLiteral("rotation and zoom"),
                                       vl::synth::rigidFlow({384, 288}, 2.0, 1.02, {1.5, -1.0}, 7));
                   }});
    out.push_back({QStringLiteral("Synthetic: flat square (aperture)"), [] {
                       return fromPair(QStringLiteral("flat square"),
                                       vl::flow::slidingSquare({384, 288}, {120, 90, 110, 90}, {3.0, 2.0}, 3));
                   }});
    // Middlebury sequences with public ground truth (Baker et al. 2011), frames 10 and 11.
    for (const char* seq : {"RubberWhale", "Hydrangea", "Dimetrodon", "Grove2", "Grove3", "Urban2", "Urban3", "Venus"}) {
        const QString s = QString::fromUtf8(seq);
        const QString f0 = dataFile(QStringLiteral("flow/other-data/%1/frame10.png").arg(s));
        const QString f1 = dataFile(QStringLiteral("flow/other-data/%1/frame11.png").arg(s));
        const QString gt = dataFile(QStringLiteral("flow/other-gt-flow/%1/flow10.flo").arg(s));
        if (f0.isEmpty() || f1.isEmpty() || gt.isEmpty()) continue;
        out.push_back({s + QStringLiteral(" (Middlebury)"), [s, f0, f1, gt] {
                           Inputs in;
                           in.name = s;
                           in.images = loadImages({f0, f1});
                           in.truth = {vl::flow::readFlo(gt.toStdString())};
                           return in;
                       }});
    }
    return out;
}

}  // namespace vl::app::flow_samples
