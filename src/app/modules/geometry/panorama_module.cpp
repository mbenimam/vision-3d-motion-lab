// Topic: homography estimation (SIFT matches, normalised DLT inside RANSAC) and panorama stitching.

#include <cmath>
#include <numeric>

#include <opencv2/imgproc.hpp>

#include "app/module.hpp"
#include "app/modules/geometry/geometry_common.hpp"
#include "app/modules/geometry/geometry_samples.hpp"
#include "core/geometry/features.hpp"
#include "core/geometry/homography.hpp"
#include "core/geometry/stitch.hpp"

namespace vl::app {
namespace {

namespace g = vl::geometry;
using namespace geometry_common;

constexpr double kConfidence[] = {0.90, 0.99, 0.999};

QPolygonF polygon(const std::vector<cv::Point2d>& pts) {
    QPolygonF poly;
    for (const auto& p : pts) poly << qp(p);
    return poly;
}

cv::Point2d topCentre(const std::vector<cv::Point2d>& outline) {
    cv::Point2d c(0, 0);
    double top = 1e300;
    for (const auto& p : outline) c += p, top = std::min(top, p.y);
    c *= 1.0 / double(std::max<size_t>(outline.size(), 1));
    return {c.x, top};
}

class PanoramaModule final : public Module {
public:
    Info info() const override {
        return {QStringLiteral("panorama"), QStringLiteral("Panorama"), QStringLiteral("Geometry"),
                QStringLiteral("Photos of a plane, or taken by a camera that only <i>turns</i>, are related by a "
                               "3x3 <b>homography</b>. Four point matches fix it (DLT); <b>RANSAC</b> draws random "
                               "4-point samples, keeps the model most matches agree with and refits it on them. "
                               "Every image is then mapped into the middle one's frame and blended."),
                QStringLiteral("on <i>Graffiti 1 → 3</i>, raise the ratio test to 0.95: twice the matches, many of "
                               "them wrong; RANSAC needs twice the trials and still lands within a few pixels of the "
                               "true H."),
                InputKind::ImagePair, 110, QStringLiteral("Panorama")};
    }

    std::vector<ParamSpec> params() const override {
        return {
            ParamSpec::real(QStringLiteral("ratio"), QStringLiteral("Ratio test (nearest / second)"), 0.5, 0.95, 0.75,
                            0.01, 2)
                .withHelp(QStringLiteral("Keep a match only when its nearest descriptor is clearly closer than the "
                                         "second nearest. Higher: more matches, more of them wrong.")),
            ParamSpec::toggle(QStringLiteral("mutual"), QStringLiteral("Mutual best match"), true)
                .withHelp(QStringLiteral("Also require the match to be the nearest in the other direction.")),
            ParamSpec::real(QStringLiteral("threshold"), QStringLiteral("Inlier threshold"), 0.5, 10, 3, 0.5, 1,
                            QStringLiteral(" px"))
                .withHelp(QStringLiteral("A match agrees with H when H maps its first point within this distance "
                                         "of its second.")),
            ParamSpec::choice(QStringLiteral("confidence"), QStringLiteral("RANSAC confidence"),
                              {QStringLiteral("90 %"), QStringLiteral("99 %"), QStringLiteral("99.9 %")}, 1)
                .withHelp(QStringLiteral("Probability of drawing at least one all-inlier sample; sets the number "
                                         "of trials N = log(1 − p) / log(1 − w⁴).")),
            ParamSpec::toggle(QStringLiteral("normalise"), QStringLiteral("Normalise coordinates"), true)
                .withHelp(QStringLiteral("Hartley: centre and scale the points before the DLT. Off, the least-"
                                         "squares refit is badly conditioned and drifts from the truth.")),
            ParamSpec::toggle(QStringLiteral("feather"), QStringLiteral("Feather blending"), true)
                .withHelp(QStringLiteral("Weigh each image by the distance to its border. Off, images are pasted "
                                         "in turn and exposure differences show as hard seams.")),
        };
    }

    std::vector<Sample> samples() const override { return geometry_samples::panorama(); }

    Output compute(const Inputs& in, const Params& p, const RunContext& ctx) const override {
        const int n = static_cast<int>(in.images.size());
        const int ref = n / 2;
        g::MatchParams mp{float(realParam(p, "ratio")), toggleParam(p, "mutual")};
        g::RansacParams rp;
        rp.threshold = realParam(p, "threshold");
        rp.confidence = kConfidence[std::clamp(intParam(p, "confidence"), 0, 2)];
        const bool normalise = toggleParam(p, "normalise");

        // 1. Match and fit each neighbouring pair i -> i+1.
        struct PairResult {
            Correspondences c;
            g::HomographyEstimate h;
        };
        std::vector<PairResult> pairs;
        for (int i = 0; i + 1 < n; ++i) {
            report(ctx.progress, 0.6 * i / std::max(1, n - 1), "matching image " + std::to_string(i + 1));
            PairResult pr;
            pr.c = correspond(cache_, in.images[size_t(i)], in.images[size_t(i + 1)], mp, 3000, ctx.stop);
            pr.h = g::estimateHomography(pr.c.a, pr.c.b, rp, normalise, ctx.stop);
            pairs.push_back(std::move(pr));
        }
        ctx.check();

        Output out;
        Frame f;

        // 2. Matches and inliers, side by side.
        const Strip strip = sideBySide(in.images);
        View matches{QStringLiteral("Matches"), toQImage(strip.image)};
        View inliers{QStringLiteral("Inliers"), toQImage(strip.image)};
        int nMatches = 0, nInliers = 0;
        bool allOk = true;
        for (int i = 0; i + 1 < n; ++i) {
            const auto& pr = pairs[size_t(i)];
            const cv::Point2d oa = strip.offsets[size_t(i)], ob = strip.offsets[size_t(i + 1)];
            addMatchLines(matches.shapes, pr.c.a, pr.c.b, oa, ob, pr.h.inliers, true);
            addMatchLines(inliers.shapes, pr.c.a, pr.c.b, oa, ob, pr.h.inliers, false);
            nMatches += int(pr.c.a.size());
            nInliers += pr.h.nInliers;
            allOk &= pr.h.ok && pr.h.nInliers >= 8;
            const QString label = QStringLiteral("%1 → %2:  %3 / %4 agree with H")
                                      .arg(i + 1).arg(i + 2).arg(pr.h.nInliers).arg(pr.c.a.size());
            matches.shapes.push_back(LabelShape{QPointF(ob.x - 12, 10), label, colors::good});
        }
        f.views = {matches, inliers};

        // 3. Panorama in the reference frame.
        if (allOk) {
            std::vector<cv::Matx33d> links;
            for (const auto& pr : pairs) links.push_back(pr.h.H);
            const auto toRef = g::chainToReference(links, ref);
            report(ctx.progress, 0.7, "warping");
            const auto pano = g::stitch(in.images, toRef, {toggleParam(p, "feather")}, ctx.stop);

            View pv{QStringLiteral("Panorama"), toQImage(pano.image)};
            for (int i = 0; i < n; ++i) {
                const bool isRef = i == ref;
                const QColor c = isRef ? colors::accent : colors::info;
                pv.shapes.push_back(PolylineShape{polygon(pano.outlines[size_t(i)]), {c, isRef ? 2.0 : 1.5}, true});
                const cv::Point2d tc = topCentre(pano.outlines[size_t(i)]);
                pv.shapes.push_back(LabelShape{QPointF(tc.x - 30, tc.y + 8),
                                               isRef ? QStringLiteral("%1 · reference").arg(i + 1) : QString::number(i + 1), c});
            }
            f.views.push_back(pv);

            // 4. Against the truth: true outlines dashed, estimated ones solid.
            const bool hasTruth = int(in.truth.size()) == n && !in.truth[0].empty();
            if (hasTruth) {
                cv::Mat dim;
                pano.image.convertTo(dim, -1, 0.45);
                View tv{QStringLiteral("Truth"), toQImage(dim)};
                const cv::Matx33d T = pano.toCanvas[size_t(ref)];  // reference frame -> canvas
                double err = 0;
                for (int i = 0; i < n; ++i) {
                    if (i == ref) continue;
                    const cv::Matx33d truth(in.truth[size_t(i)]);
                    const double e = g::cornerError(toRef[size_t(i)], truth, in.images[size_t(i)].size());
                    err += e / (n - 1);
                    tv.shapes.push_back(PolylineShape{polygon(g::warpedOutline(in.images[size_t(i)].size(), T * truth)),
                                                      {colors::info, 4.0, true}, true});
                    tv.shapes.push_back(PolylineShape{polygon(pano.outlines[size_t(i)]), {colors::accent, 1.5}, true});
                    const cv::Point2d tc = topCentre(pano.outlines[size_t(i)]);
                    tv.shapes.push_back(LabelShape{QPointF(tc.x - 60, tc.y + 8),
                                                   QStringLiteral("image %1: corners off by %2 px").arg(i + 1).arg(fmt(e, 2)),
                                                   colors::accent});
                }
                tv.shapes.push_back(PolylineShape{polygon(pano.outlines[size_t(ref)]), {colors::muted, 1.0}, true});
                tv.shapes.push_back(LabelShape{QPointF(12, 12), QStringLiteral("dashed: true H   ·   solid: estimated H"),
                                               colors::info});
                f.views.push_back(tv);
                out.stats.push_back({QStringLiteral("Corner error vs true H"), fmt(err, 2) + QStringLiteral(" px"), true});
            }
            if (pano.scale < 1.0)
                out.note = QStringLiteral("The estimated maps stretch the canvas a lot: it is shown at %1 % scale.")
                               .arg(int(100 * pano.scale));
            out.stats.push_back({QStringLiteral("Canvas"), QStringLiteral("%1 × %2 px").arg(pano.image.cols).arg(pano.image.rows)});
        } else {
            out.note = QStringLiteral("Too few matches agree on a homography: lower the ratio test or raise the "
                                      "threshold.");
        }

        // Stats and the RANSAC curve of each pair.
        const bool emphasiseInliers = out.stats.empty() || !out.stats.front().emphasis;
        out.stats.insert(out.stats.begin(),
                         {{QStringLiteral("Matches kept"), QString::number(nMatches)},
                          {QStringLiteral("Inliers"),
                           QStringLiteral("%1  (%2 %)").arg(nInliers).arg(fmt(100.0 * nInliers / std::max(1, nMatches), 0)),
                           emphasiseInliers}});
        QStringList trials, rms;
        for (const auto& pr : pairs) {
            trials << QString::number(pr.h.report.trials);
            rms << fmt(pr.h.rms, 2);
        }
        out.stats.insert(out.stats.begin() + 2, {QStringLiteral("RANSAC trials"), trials.join(QStringLiteral(" + "))});
        out.stats.insert(out.stats.begin() + 3,
                         {QStringLiteral("Transfer RMS on inliers"), rms.join(QStringLiteral(" · ")) + QStringLiteral(" px")});
        for (int i = 0; i + 1 < n; ++i) {
            const auto& c = pairs[size_t(i)].h.report.bestInliersPerTrial;
            out.series.push_back({QStringLiteral("%1 → %2: best inliers per trial").arg(i + 1).arg(i + 2),
                                  std::vector<double>(c.begin(), c.end())});
        }
        if (out.note.isEmpty() && !normalise)
            out.note = QStringLiteral("Normalisation off: the refit on all inliers works on raw pixel coordinates.");
        out.frames.push_back(std::move(f));
        return out;
    }

private:
    mutable vl::geometry::FeatureCache cache_;
};

}  // namespace

VL_REGISTER_MODULE(PanoramaModule);

}  // namespace vl::app
