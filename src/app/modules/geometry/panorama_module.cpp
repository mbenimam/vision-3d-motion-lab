// Topic: homography estimation (our SIFT and matcher, normalised DLT inside RANSAC) and panorama
// stitching, compared with OpenCV's SIFT + BFMatcher + findHomography.

#include <algorithm>
#include <cmath>
#include <numeric>

#include "app/module.hpp"
#include "app/modules/geometry/geometry_common.hpp"
#include "app/modules/geometry/geometry_samples.hpp"
#include "core/geometry/homography.hpp"
#include "core/geometry/matching.hpp"
#include "core/geometry/opencv_baselines.hpp"
#include "core/geometry/sift.hpp"
#include "core/geometry/stitch.hpp"

namespace vl::app {
namespace {

namespace g = vl::geometry;
using namespace geometry_common;

constexpr double kConfidence[] = {0.90, 0.99, 0.999};
constexpr int kMaxFeatures = 3000;

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

// One neighbouring pair i -> i+1: its correspondences and the homography fitted to them.
struct PairResult {
    g::Points a, b;
    g::HomographyEstimate h;
};

// The settings shared by ours and the baseline, read once from the parameters.
struct Settings {
    g::SiftParams sift;
    g::MatchParams match;
    g::RansacParams ransac;
    bool normalise = true, feather = true;
};

// Mean corner error of the maps image i -> reference against the truth (pixels), or -1 without truth.
double cornerErrorVsTruth(const Inputs& in, const std::vector<cv::Matx33d>& toRef, int ref) {
    const int n = int(in.images.size());
    if (int(in.truth.size()) != n || in.truth[0].empty()) return -1;
    double err = 0;
    for (int i = 0; i < n; ++i)
        if (i != ref) err += g::cornerError(toRef[size_t(i)], cv::Matx33d(in.truth[size_t(i)]), in.images[size_t(i)].size());
    return err / (n - 1);
}

// The panorama of a set of images given the links i -> i+1, with each image's outline and number.
View panoramaView(const QString& name, const std::vector<cv::Mat>& images, const std::vector<cv::Matx33d>& toRef,
                  int ref, bool feather, std::stop_token stop, g::Panorama* keep = nullptr) {
    const auto pano = g::stitch(images, toRef, {feather}, stop);
    View pv{name, toQImage(pano.image)};
    for (int i = 0; i < int(images.size()); ++i) {
        const bool isRef = i == ref;
        const QColor c = isRef ? colors::accent : colors::info;
        pv.shapes.push_back(PolylineShape{polygon(pano.outlines[size_t(i)]), {c, isRef ? 2.0 : 1.5}, true});
        const cv::Point2d tc = topCentre(pano.outlines[size_t(i)]);
        pv.shapes.push_back(LabelShape{QPointF(tc.x - 30, tc.y + 8),
                                       isRef ? QStringLiteral("%1 · reference").arg(i + 1) : QString::number(i + 1), c});
    }
    if (keep) *keep = pano;
    return pv;
}

class PanoramaModule final : public Module {
public:
    Info info() const override {
        return {QStringLiteral("panorama"), QStringLiteral("Panorama"), QStringLiteral("Geometry"),
                QStringLiteral("Photos of a plane, or taken by a camera that only <i>turns</i>, are related by a "
                               "3x3 <b>homography</b>. <b>SIFT</b> keypoints are matched between neighbours; four "
                               "matches fix H (DLT), and <b>RANSAC</b> keeps the H most matches agree with. Every "
                               "image is then mapped into the middle one's frame and blended."),
                QStringLiteral("on <i>Graffiti 1 → 3</i>, raise the ratio test to 0.95: many more matches, most of "
                               "the new ones wrong; RANSAC needs more trials and still lands within a few pixels of "
                               "the true H."),
                InputKind::ImagePair, 110, QStringLiteral("Panorama"),
                QStringLiteral("cv::SIFT + cv::BFMatcher + cv::findHomography"),
                {QStringLiteral("src/core/geometry/sift.hpp"), QStringLiteral("src/core/geometry/sift.cpp"),
                 QStringLiteral("src/core/geometry/matching.hpp"), QStringLiteral("src/core/geometry/matching.cpp"),
                 QStringLiteral("src/core/geometry/ransac.hpp"), QStringLiteral("src/core/geometry/ransac.cpp"),
                 QStringLiteral("src/core/geometry/homography.hpp"), QStringLiteral("src/core/geometry/homography.cpp"),
                 QStringLiteral("src/core/geometry/stitch.hpp"), QStringLiteral("src/core/geometry/stitch.cpp")}};
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
            ParamSpec::toggle(QStringLiteral("upsample"), QStringLiteral("SIFT on the image doubled"), true)
                .withHelp(QStringLiteral("Lowe's first step: double the image before the scale space. Off: about "
                                         "a third of the keypoints (the small ones are lost), four times faster.")),
        };
    }

    std::vector<Sample> samples() const override { return geometry_samples::panorama(); }

    Output compute(const Inputs& in, const Params& p, const RunContext& ctx) const override {
        const int n = int(in.images.size());
        const int ref = n / 2;
        Settings s;
        s.sift.maxFeatures = kMaxFeatures;
        s.sift.upsample = toggleParam(p, "upsample");
        s.match = {float(realParam(p, "ratio")), toggleParam(p, "mutual")};
        s.ransac.threshold = realParam(p, "threshold");
        s.ransac.confidence = kConfidence[std::clamp(intParam(p, "confidence"), 0, 2)];
        s.normalise = toggleParam(p, "normalise");
        s.feather = toggleParam(p, "feather");

        // 1. SIFT on every image (in parallel), then matches and a homography for each pair i -> i+1.
        report(ctx.progress, 0.05, "SIFT");
        std::vector<double> detectMs;
        const std::vector<g::Features> features = detectAll(cache_, in.images, s.sift, ctx.stop, detectMs);
        double oursMs = *std::max_element(detectMs.begin(), detectMs.end());  // wall time of the parallel detection
        std::vector<PairResult> pairs;
        for (int i = 0; i + 1 < n; ++i) {
            report(ctx.progress, 0.4 + 0.3 * i / std::max(1, n - 1), "matching image " + std::to_string(i + 1));
            const Correspondences c = correspond(features[size_t(i)], features[size_t(i + 1)], s.match, ctx.stop);
            const Stopwatch sw;
            pairs.push_back({c.a, c.b, g::estimateHomography(c.a, c.b, s.ransac, s.normalise, ctx.stop)});
            oursMs += c.matchMs + sw.ms();
        }
        ctx.check();

        Output out;
        Frame f;
        const Strip strip = sideBySide(in.images);

        // 2. The keypoints, then the matches and the inliers, side by side.
        View keypoints{QStringLiteral("Keypoints"), toQImage(strip.image)};
        int nKeypoints = 0;
        for (int i = 0; i < n; ++i) {
            addKeypoints(keypoints.shapes, features[size_t(i)].keypoints, strip.offsets[size_t(i)]);
            nKeypoints += int(features[size_t(i)].keypoints.size());
        }
        keypoints.shapes.push_back(LabelShape{QPointF(12, 10),
                                              QStringLiteral("circle: 2σ · tick: orientation · strongest 400 per image"),
                                              colors::accent});
        View matches{QStringLiteral("Matches"), toQImage(strip.image)};
        View inliers{QStringLiteral("Inliers"), toQImage(strip.image)};
        int nMatches = 0, nInliers = 0;
        bool allOk = true;
        for (int i = 0; i + 1 < n; ++i) {
            const auto& pr = pairs[size_t(i)];
            const cv::Point2d oa = strip.offsets[size_t(i)], ob = strip.offsets[size_t(i + 1)];
            addMatchLines(matches.shapes, pr.a, pr.b, oa, ob, pr.h.inliers, true);
            addMatchLines(inliers.shapes, pr.a, pr.b, oa, ob, pr.h.inliers, false);
            nMatches += int(pr.a.size());
            nInliers += pr.h.nInliers;
            allOk = allOk && pr.h.ok && pr.h.nInliers >= 8;
            matches.shapes.push_back(LabelShape{QPointF(ob.x - 12, 10),
                                                QStringLiteral("%1 → %2:  %3 / %4 agree with H")
                                                    .arg(i + 1).arg(i + 2).arg(pr.h.nInliers).arg(pr.a.size()),
                                                colors::good});
        }
        f.views = {keypoints, matches, inliers};

        // 3. The panorama in the reference frame, and 4. against the truth.
        std::vector<cv::Matx33d> toRef;
        double err = -1;
        if (allOk) {
            std::vector<cv::Matx33d> links;
            for (const auto& pr : pairs) links.push_back(pr.h.H);
            toRef = g::chainToReference(links, ref);
            report(ctx.progress, 0.7, "warping");
            g::Panorama pano;
            f.views.push_back(panoramaView(QStringLiteral("Panorama"), in.images, toRef, ref, s.feather, ctx.stop, &pano));
            err = cornerErrorVsTruth(in, toRef, ref);
            if (err >= 0) f.views.push_back(truthView(in, pano, toRef, ref));
            if (err >= 0) out.stats.push_back({QStringLiteral("Corner error vs true H"), fmt(err, 2) + QStringLiteral(" px"), true});
            if (pano.scale < 1.0)
                out.note = QStringLiteral("The estimated maps stretch the canvas a lot: it is shown at %1 % scale.")
                               .arg(int(100 * pano.scale));
            out.stats.push_back({QStringLiteral("Canvas"), QStringLiteral("%1 × %2 px").arg(pano.image.cols).arg(pano.image.rows)});
        } else {
            out.note = QStringLiteral("Too few matches agree on a homography: lower the ratio test or raise the "
                                      "threshold.");
        }

        // Stats and the RANSAC curve of each pair.
        const bool emphasiseInliers = err < 0;
        QStringList trials, rms;
        for (const auto& pr : pairs) {
            trials << QString::number(pr.h.report.trials);
            rms << fmt(pr.h.rms, 2);
        }
        out.stats.insert(out.stats.begin(),
                         {{QStringLiteral("Keypoints · matches"), QStringLiteral("%1 · %2").arg(nKeypoints).arg(nMatches)},
                          {QStringLiteral("Inliers"),
                           QStringLiteral("%1  (%2 %)").arg(nInliers).arg(fmt(100.0 * nInliers / std::max(1, nMatches), 0)),
                           emphasiseInliers},
                          {QStringLiteral("RANSAC trials"), trials.join(QStringLiteral(" + "))},
                          {QStringLiteral("Transfer RMS on inliers"), rms.join(QStringLiteral(" · ")) + QStringLiteral(" px")}});
        for (int i = 0; i + 1 < n; ++i) {
            const auto& c = pairs[size_t(i)].h.report.bestInliersPerTrial;
            out.series.push_back({QStringLiteral("%1 → %2: best inliers per trial").arg(i + 1).arg(i + 2),
                                  std::vector<double>(c.begin(), c.end())});
        }
        if (out.note.isEmpty() && !s.normalise)
            out.note = QStringLiteral("Normalisation off: the refit on all inliers works on raw pixel coordinates.");

        if (ctx.compare) {
            report(ctx.progress, 0.85, "OpenCV baseline");
            compareWithOpenCv(in, s, pairs, toRef, strip, nKeypoints, oursMs, err, f, out, ctx);
        }
        out.frames.push_back(std::move(f));
        return out;
    }

private:
    // Dimmed panorama with the true outlines (dashed) over the estimated ones (solid).
    static View truthView(const Inputs& in, const g::Panorama& pano, const std::vector<cv::Matx33d>& toRef, int ref) {
        cv::Mat dim;
        pano.image.convertTo(dim, -1, 0.45);
        View tv{QStringLiteral("Truth"), toQImage(dim)};
        const cv::Matx33d T = pano.toCanvas[size_t(ref)];  // reference frame -> canvas
        for (int i = 0; i < int(in.images.size()); ++i) {
            if (i == ref) continue;
            const cv::Matx33d truth(in.truth[size_t(i)]);
            const double e = g::cornerError(toRef[size_t(i)], truth, in.images[size_t(i)].size());
            tv.shapes.push_back(PolylineShape{polygon(g::warpedOutline(in.images[size_t(i)].size(), T * truth)),
                                              {colors::info, 4.0, true}, true});
            tv.shapes.push_back(PolylineShape{polygon(pano.outlines[size_t(i)]), {colors::accent, 1.5}, true});
            const cv::Point2d tc = topCentre(pano.outlines[size_t(i)]);
            tv.shapes.push_back(LabelShape{QPointF(tc.x - 60, tc.y + 8),
                                           QStringLiteral("image %1: corners off by %2 px").arg(i + 1).arg(fmt(e, 2)),
                                           colors::accent});
        }
        tv.shapes.push_back(PolylineShape{polygon(pano.outlines[size_t(ref)]), {colors::muted, 1.0}, true});
        tv.shapes.push_back(LabelShape{QPointF(12, 12), QStringLiteral("dashed: true H   ·   solid: estimated H"), colors::info});
        return tv;
    }

    // (a) The whole OpenCV pipeline (cv::SIFT, BFMatcher 2-NN + ratio test, findHomography RANSAC) on the
    // same images; (b) cv::findHomography on *our* matches, to compare the estimators on identical data.
    static void compareWithOpenCv(const Inputs& in, const Settings& s, const std::vector<PairResult>& ours,
                                  const std::vector<cv::Matx33d>& oursToRef, const Strip& strip, int oursKeypoints,
                                  double oursMs, double oursErr, Frame& f, Output& out, const RunContext& ctx) {
        const int n = int(in.images.size()), ref = n / 2;
        View matches{QStringLiteral("OpenCV · Matches"), toQImage(strip.image)};
        std::vector<cv::Matx33d> links, sameMatchLinks;
        int keypoints = 0, nMatches = 0, nInliers = 0, oursInliers = 0, sameInliers = 0;
        double cvMs = 0, rmsOurs = 0, rmsCv = 0;
        bool ok = true;
        for (int i = 0; i + 1 < n; ++i) {
            ctx.check();
            const auto c = g::opencv::siftMatches(in.images[size_t(i)], in.images[size_t(i + 1)], s.match.ratio, kMaxFeatures);
            const auto h = g::opencv::findHomography(c.a, c.b, s.ransac.threshold, s.ransac.confidence);
            keypoints += (i == 0 ? c.keypointsA : 0) + c.keypointsB;
            cvMs += c.ms + h.ms;
            nMatches += int(c.a.size());
            nInliers += h.nInliers;
            ok = ok && h.ok;
            if (!c.a.empty()) addMatchLines(matches.shapes, c.a, c.b, strip.offsets[size_t(i)], strip.offsets[size_t(i + 1)],
                                            h.inliers.empty() ? std::vector<std::uint8_t>(c.a.size(), 0) : h.inliers, true);
            links.push_back(h.H);
            std::vector<double> e;
            g::transferErrors(h.H, c.a, c.b, e);
            rmsCv += inlierRms(e, h.inliers) / (n - 1);
            rmsOurs += ours[size_t(i)].h.rms / (n - 1);
            // (b) the same estimator question on our matches.
            const auto same = g::opencv::findHomography(ours[size_t(i)].a, ours[size_t(i)].b, s.ransac.threshold,
                                                        s.ransac.confidence);
            sameMatchLinks.push_back(same.H);
            sameInliers += same.nInliers;
            oursInliers += ours[size_t(i)].h.nInliers;
        }
        f.views.push_back(matches);
        double cvErr = -1;
        if (ok) {
            const auto toRef = g::chainToReference(links, ref);
            View pv = panoramaView(QStringLiteral("OpenCV · Panorama"), in.images, toRef, ref, s.feather, ctx.stop);
            pv.shapes.push_back(LabelShape{QPointF(12, 12), QStringLiteral("OpenCV's homographies, warped by our stitcher"),
                                           colors::info});
            f.views.push_back(pv);
            cvErr = cornerErrorVsTruth(in, toRef, ref);
        }

        auto& cmp = out.comparison;
        if (oursErr >= 0 && cvErr >= 0)
            cmp.push_back({QStringLiteral("Corner error vs true H"), oursErr, cvErr, QStringLiteral(" px"), 2});
        cmp.push_back({QStringLiteral("Keypoints"), double(oursKeypoints), double(keypoints), {}, 0, false});
        cmp.push_back({QStringLiteral("Matches (ratio test)"), double(countMatches(ours)),
                       double(nMatches), {}, 0, false});
        cmp.push_back({QStringLiteral("Inliers"), double(oursInliers), double(nInliers), {}, 0, false});
        cmp.push_back({QStringLiteral("Transfer RMS on inliers"), rmsOurs, rmsCv, QStringLiteral(" px"), 2});
        // (b) Same matches, two RANSAC implementations.
        const double sameErr = oursToRef.empty() ? -1 : cornerErrorVsTruth(in, g::chainToReference(sameMatchLinks, ref), ref);
        if (oursErr >= 0 && sameErr >= 0)
            cmp.push_back({QStringLiteral("Our matches: corner error (vs findHomography)"), oursErr, sameErr,
                           QStringLiteral(" px"), 2});
        cmp.push_back({QStringLiteral("Our matches: inliers (vs findHomography)"), double(oursInliers), double(sameInliers),
                       {}, 0, false});
        cmp.push_back({QStringLiteral("Run time"), oursMs, cvMs, QStringLiteral(" ms"), 0});
    }

    static int countMatches(const std::vector<PairResult>& pairs) {
        int m = 0;
        for (const auto& pr : pairs) m += int(pr.a.size());
        return m;
    }

    static double inlierRms(const std::vector<double>& e, const std::vector<std::uint8_t>& inliers) {
        double sum = 0;
        int k = 0;
        for (size_t i = 0; i < e.size() && i < inliers.size(); ++i)
            if (inliers[i]) sum += e[i] * e[i], ++k;
        return k ? std::sqrt(sum / k) : 0.0;
    }

    mutable FeatureCache cache_;
};

}  // namespace

VL_REGISTER_MODULE(PanoramaModule);

}  // namespace vl::app
