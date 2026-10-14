// Topic: epipolar geometry. The fundamental matrix by the normalised 8-point algorithm inside RANSAC on
// our SIFT matches, its epipolar lines and epipoles, click-to-see-the-line between the two views, and
// cv::findFundamentalMat on the same matches as the baseline.

#include <cmath>

#include "app/module.hpp"
#include "app/modules/geometry/geometry_common.hpp"
#include "app/modules/geometry/geometry_samples.hpp"
#include "core/geometry/fundamental.hpp"
#include "core/geometry/homography.hpp"
#include "core/geometry/opencv_baselines.hpp"
#include "core/geometry/sift.hpp"

namespace vl::app {
namespace {

namespace g = vl::geometry;
using namespace geometry_common;

constexpr int kLines = 14;       // epipolar lines drawn per view
constexpr int kMatchesView = 0, kLeftView = 1, kRightView = 2;

QString pointText(const std::optional<cv::Point2d>& p) {
    if (!p || std::abs(p->x) > 1e5 || std::abs(p->y) > 1e5) return QStringLiteral("at infinity");
    return QStringLiteral("(%1, %2)").arg(p->x, 0, 'f', 0).arg(p->y, 0, 'f', 0);
}

void markEpipole(View& v, const std::optional<cv::Point2d>& e, cv::Size size, const QString& name) {
    if (!e || e->x < 0 || e->y < 0 || e->x >= size.width || e->y >= size.height) return;
    v.shapes.push_back(EllipseShape{qp(*e), 9, 9, 0, {colors::accent, 2.0}});
    v.shapes.push_back(PointsShape{QPolygonF{qp(*e)}, colors::accent, 3.0});
    v.shapes.push_back(LabelShape{qp(*e) + QPointF(12, -24), name, colors::accent});
}

class EpipolarModule final : public Module {
public:
    Info info() const override {
        return {QStringLiteral("epipolar"), QStringLiteral("Epipolar geometry"), QStringLiteral("Geometry"),
                QStringLiteral("A point seen in one image must lie, in the other, on a line: the image of its viewing "
                               "ray. The <b>fundamental matrix</b> F encodes this (x₂ᵀ F x₁ = 0); all those lines meet at "
                               "the <b>epipole</b>, the image of the other camera's centre. F comes from 8 matches "
                               "(8-point algorithm), inside RANSAC to survive the wrong ones."),
                QStringLiteral("click a point of the left image of <i>Merton 001 / 002</i> and follow its line on the "
                               "right; then, on the synthetic room, switch off <i>Enforce rank 2</i>: σ₃ leaves 0, the "
                               "epipole is gone and the error against the true F doubles."),
                InputKind::ImagePair, 120, QStringLiteral("Left"), QStringLiteral("cv::findFundamentalMat (FM_RANSAC)"),
                {QStringLiteral("src/core/geometry/fundamental.hpp"), QStringLiteral("src/core/geometry/fundamental.cpp"),
                 QStringLiteral("src/core/geometry/ransac.hpp"), QStringLiteral("src/core/geometry/ransac.cpp"),
                 QStringLiteral("src/core/geometry/homography.hpp"), QStringLiteral("src/core/geometry/homography.cpp"),
                 QStringLiteral("src/core/geometry/sift.hpp"), QStringLiteral("src/core/geometry/sift.cpp"),
                 QStringLiteral("src/core/geometry/matching.hpp"), QStringLiteral("src/core/geometry/matching.cpp")}};
    }

    std::vector<ParamSpec> params() const override {
        return {
            ParamSpec::real(QStringLiteral("ratio"), QStringLiteral("Ratio test (nearest / second)"), 0.5, 0.95, 0.8,
                            0.01, 2)
                .withHelp(QStringLiteral("Keep a match only when its nearest descriptor is clearly closer than the "
                                         "second nearest.")),
            ParamSpec::real(QStringLiteral("threshold"), QStringLiteral("Inlier threshold (Sampson)"), 0.25, 5, 1.0,
                            0.25, 2, QStringLiteral(" px"))
                .withHelp(QStringLiteral("A match agrees with F when its Sampson distance, a first-order estimate of "
                                         "how far the points must move to satisfy x₂ᵀ F x₁ = 0, is below this.")),
            ParamSpec::choice(QStringLiteral("confidence"), QStringLiteral("RANSAC confidence"),
                              {QStringLiteral("90 %"), QStringLiteral("99 %"), QStringLiteral("99.9 %")}, 1)
                .withHelp(QStringLiteral("Sets the number of trials N = log(1 − p) / log(1 − w⁸): eight points per "
                                         "sample make clean samples much rarer than for a homography.")),
            ParamSpec::toggle(QStringLiteral("normalise"), QStringLiteral("Normalise coordinates"), true)
                .withHelp(QStringLiteral("Hartley: centre and scale the points before solving. Off, entries of the "
                                         "linear system span six orders of magnitude and F loses accuracy.")),
            ParamSpec::toggle(QStringLiteral("rank2"), QStringLiteral("Enforce rank 2"), true)
                .withHelp(QStringLiteral("Zero the smallest singular value of F. Off, F is full rank: there is no "
                                         "epipole, and the lines no longer meet exactly in one point.")),
            ParamSpec::toggle(QStringLiteral("upsample"), QStringLiteral("SIFT on the image doubled"), true)
                .withHelp(QStringLiteral("Lowe's first step: double the image before the scale space. Off: fewer "
                                         "keypoints (the small ones are lost), four times faster.")),
        };
    }

    std::vector<Sample> samples() const override { return geometry_samples::epipolar(); }

    Output compute(const Inputs& in, const Params& p, const RunContext& ctx) const override {
        static constexpr double kConfidence[] = {0.90, 0.99, 0.999};
        const cv::Mat& L = in.images[0];
        const cv::Mat& R = in.images[1];
        report(ctx.progress, 0.1, "matching");
        g::SiftParams sp;
        sp.upsample = toggleParam(p, "upsample");
        std::vector<double> detectMs;
        const auto features = detectAll(cache_, {L, R}, sp, ctx.stop, detectMs);
        const auto c = correspond(features[0], features[1], {float(realParam(p, "ratio")), true}, ctx.stop);

        g::RansacParams rp;
        rp.threshold = realParam(p, "threshold");
        rp.confidence = kConfidence[std::clamp(intParam(p, "confidence"), 0, 2)];
        const g::EightPointOptions opt{toggleParam(p, "normalise"), toggleParam(p, "rank2")};
        report(ctx.progress, 0.6, "RANSAC");
        const Stopwatch ransacTime;
        const auto e = g::estimateFundamental(c.a, c.b, rp, opt, ctx.stop);
        const double oursMs = ransacTime.ms();
        ctx.check();

        Output out;
        Frame f;
        const Strip strip = sideBySide({L, R});
        View matches{QStringLiteral("Matches"), toQImage(strip.image)};
        const cv::Point2d offR = strip.offsets[1];
        if (!c.a.empty()) addMatchLines(matches.shapes, c.a, c.b, {0, 0}, offR, e.inliers, true);
        View left{QStringLiteral("Left"), toQImage(L)}, right{QStringLiteral("Right"), toQImage(R)};
        const QString hint = QStringLiteral("click: its epipolar line in the other image");
        left.probe = [hint](QPoint q) { return QStringLiteral("(%1, %2)  ·  %3").arg(q.x()).arg(q.y()).arg(hint); };
        right.probe = left.probe;

        if (!e.ok) {
            out.stats.push_back({QStringLiteral("Matches (ratio + mutual)"), QString::number(c.a.size())});
            out.note = QStringLiteral("Fewer than 8 usable matches: F cannot be estimated. Raise the ratio test.");
            f.views = {matches, left, right};
            out.frames.push_back(std::move(f));
            return out;
        }

        // F rides along with the views so a click can draw lines without recomputing anything.
        const cv::Mat F = cv::Mat(e.F).clone();
        matches.data = left.data = right.data = F;

        // A few well-spread inliers: each point and its epipolar line in the other view share a colour.
        const auto pick = spreadSubset(c.a, e.inliers, kLines);
        for (int k = 0; k < static_cast<int>(pick.size()); ++k) {
            const size_t i = size_t(pick[size_t(k)]);
            const QColor col = pairColor(k);
            if (auto s = lineShape(g::lineInImage1(e.F, c.b[i]), L.size(), {0, 0}, {col, 1.4})) left.shapes.push_back(*s);
            if (auto s = lineShape(g::lineInImage2(e.F, c.a[i]), R.size(), {0, 0}, {col, 1.4})) right.shapes.push_back(*s);
            left.shapes.push_back(PointsShape{QPolygonF{qp(c.a[i])}, col, 4.0});
            right.shapes.push_back(PointsShape{QPolygonF{qp(c.b[i])}, col, 4.0});
        }
        const auto ep = g::epipoles(e.F);
        const auto e1 = g::finitePoint(ep.e1), e2 = g::finitePoint(ep.e2);
        if (opt.enforceRank2) {
            markEpipole(left, e1, L.size(), QStringLiteral("epipole e₁"));
            markEpipole(right, e2, R.size(), QStringLiteral("epipole e₂"));
        }
        f.views = {matches, left, right};

        // Against the truth: estimated lines (solid) vs true lines (dashed) for true correspondences.
        const bool hasTruth = in.truth.size() >= 2 && !in.truth[0].empty();
        double truthErr = 0;
        g::Points t1, t2;
        if (hasTruth) {
            const cv::Matx33d Ft(in.truth[0]);
            const cv::Mat& corr = in.truth[1];
            for (int i = 0; i < corr.rows; ++i)
                t1.emplace_back(corr.at<double>(i, 0), corr.at<double>(i, 1)), t2.emplace_back(corr.at<double>(i, 2), corr.at<double>(i, 3));
            truthErr = g::meanSymmetricEpipolarDistance(e.F, t1, t2);
            View tv{QStringLiteral("Truth"), toQImage(R), {}, right.probe, F};
            const auto tp = spreadSubset(t1, std::vector<std::uint8_t>(t1.size(), 1), 10);
            for (int k = 0; k < static_cast<int>(tp.size()); ++k) {
                const size_t i = size_t(tp[size_t(k)]);
                if (auto s = lineShape(g::lineInImage2(Ft, t1[i]), R.size(), {0, 0}, {colors::info, 2.0, true}))
                    tv.shapes.push_back(*s);
                if (auto s = lineShape(g::lineInImage2(e.F, t1[i]), R.size(), {0, 0}, {colors::accent, 1.2}))
                    tv.shapes.push_back(*s);
                tv.shapes.push_back(PointsShape{QPolygonF{qp(t2[i])}, colors::good, 3.5});
            }
            tv.shapes.push_back(LabelShape{QPointF(12, 12),
                                           QStringLiteral("dashed: true F   ·   solid: estimated F   ·   dots: true matches"),
                                           colors::info});
            f.views.push_back(tv);
        }

        // Is the scene planar? Then one homography explains the matches and F is not determined: any
        // F = [e₂]× H fits, for every e₂.
        g::RansacParams hp = rp;
        hp.threshold = std::max(2.0, 2 * rp.threshold);
        const auto h = g::estimateHomography(c.a, c.b, hp, true, ctx.stop);
        int both = 0;
        for (size_t i = 0; i < c.a.size(); ++i) both += e.inliers[i] && h.inliers[i];
        const double planarShare = double(both) / std::max(1, e.nInliers);

        const double sigmaRatio = e.singularValues[2] / std::max(e.singularValues[1], 1e-300);
        out.stats.push_back({QStringLiteral("Keypoints"), QStringLiteral("%1 · %2").arg(features[0].keypoints.size())
                                                              .arg(features[1].keypoints.size())});
        out.stats.push_back({QStringLiteral("Inliers / matches"),
                             QStringLiteral("%1 / %2").arg(e.nInliers).arg(c.a.size()),
                             !hasTruth});
        out.stats.push_back({QStringLiteral("Mean Sampson error"), fmt(e.meanSampson, 2) + QStringLiteral(" px")});
        if (hasTruth)
            out.stats.push_back({QStringLiteral("Error vs true F"), fmt(truthErr, 2) + QStringLiteral(" px"), true});
        out.stats.push_back({QStringLiteral("Rank (σ₃ / σ₂)"),
                             sigmaRatio < 1e-9 ? QStringLiteral("2  (0)") : QStringLiteral("3  (%1)").arg(sigmaRatio, 0, 'g', 2)});
        if (opt.enforceRank2) {
            out.stats.push_back({QStringLiteral("Epipole e₁ (left)"), pointText(e1)});
            out.stats.push_back({QStringLiteral("Epipole e₂ (right)"), pointText(e2)});
        }
        const auto& curve = e.report.bestInliersPerTrial;
        out.series.push_back({QStringLiteral("Best inliers per trial (%1)").arg(e.report.trials),
                              std::vector<double>(curve.begin(), curve.end())});

        if (planarShare > 0.7)
            out.note = QStringLiteral("%1 % of the inliers fit a single homography: the scene is (nearly) a plane, "
                                      "so F is ill-posed. Every F = [e₂]× H explains them; the epipoles are arbitrary.")
                           .arg(fmt(100 * planarShare, 0));
        else if (!opt.enforceRank2)
            out.note = QStringLiteral("Rank 2 not enforced: F has no exact null vector, so its epipolar lines only nearly meet.");
        else if (!opt.normalise)
            out.note = QStringLiteral("Normalisation off: the 8-point system mixes terms of order 1 and 10⁶.");
        if (ctx.compare) compareWithOpenCv(L, c, e, rp, oursMs, hasTruth ? &t1 : nullptr, &t2, truthErr, f, out);
        out.frames.push_back(std::move(f));
        return out;
    }

    // cv::findFundamentalMat (FM_RANSAC) on *our* matches, same threshold and confidence: the two
    // estimators answer the same question on identical data. Its epipolar lines on the left image.
    static void compareWithOpenCv(const cv::Mat& L, const Correspondences& c, const g::FundamentalEstimate& ours,
                                  const g::RansacParams& rp, double oursMs, const g::Points* t1, const g::Points* t2,
                                  double oursTruthErr, Frame& f, Output& out) {
        const auto cvF = g::opencv::findFundamental(c.a, c.b, rp.threshold, rp.confidence);
        View left{QStringLiteral("OpenCV · Left"), toQImage(L)};
        double sampson = 0;
        if (cvF.ok) {
            const auto pick = spreadSubset(c.a, cvF.inliers, kLines);
            for (int k = 0; k < int(pick.size()); ++k) {
                const size_t i = size_t(pick[size_t(k)]);
                const QColor col = pairColor(k);
                if (auto s = lineShape(g::lineInImage1(cvF.F, c.b[i]), L.size(), {0, 0}, {col, 1.4})) left.shapes.push_back(*s);
                left.shapes.push_back(PointsShape{QPolygonF{qp(c.a[i])}, col, 4.0});
            }
            markEpipole(left, g::finitePoint(g::epipoles(cvF.F).e1), L.size(), QStringLiteral("epipole e₁"));
            for (size_t i = 0; i < c.a.size(); ++i)
                if (cvF.inliers[i]) sampson += g::sampsonDistance(cvF.F, c.a[i], c.b[i]);
            sampson /= std::max(1, cvF.nInliers);
        } else {
            left.shapes.push_back(LabelShape{QPointF(12, 12), QStringLiteral("OpenCV found no F"), colors::bad});
        }
        left.shapes.push_back(LabelShape{QPointF(12, double(L.rows) - 28),
                                         QStringLiteral("cv::findFundamentalMat on our matches: %1 inliers").arg(cvF.nInliers),
                                         colors::info});
        f.views.push_back(left);

        auto& cmp = out.comparison;
        if (t1 && cvF.ok)
            cmp.push_back({QStringLiteral("Error vs true F"), oursTruthErr, g::meanSymmetricEpipolarDistance(cvF.F, *t1, *t2),
                           QStringLiteral(" px"), 2});
        cmp.push_back({QStringLiteral("Inliers (same matches)"), double(ours.nInliers), double(cvF.nInliers), {}, 0, false});
        cmp.push_back({QStringLiteral("Mean Sampson error on inliers"), ours.meanSampson, sampson, QStringLiteral(" px"), 2});
        cmp.push_back({QStringLiteral("Run time (RANSAC)"), oursMs, cvF.ms, QStringLiteral(" ms"), 1});
    }

    // Click a point in one image: its epipolar line in the other (and on the side-by-side view).
    std::vector<ClickShapes> onClick(const Output& out, int, int view, QPointF px) const override {
        if (out.frames.empty() || out.frames.front().views.size() < 3) return {};
        const auto& views = out.frames.front().views;
        const cv::Mat& Fm = views[size_t(kLeftView)].data;
        if (Fm.empty()) return {};
        const cv::Matx33d F(Fm);
        const cv::Size sl(views[size_t(kLeftView)].image.width(), views[size_t(kLeftView)].image.height());
        const cv::Size sr(views[size_t(kRightView)].image.width(), views[size_t(kRightView)].image.height());
        const double offR = views[size_t(kMatchesView)].image.width() - sr.width;

        // Which image was clicked, in its own coordinates.
        cv::Point2d q(px.x(), px.y());
        bool fromLeft = view == kLeftView;
        if (view == kMatchesView) {
            fromLeft = q.x < sl.width;
            if (!fromLeft) q.x -= offR;
        } else if (view != kLeftView && view != kRightView) {
            if (views[size_t(view)].name != QStringLiteral("Truth")) return {};  // the OpenCV view has its own F
            fromLeft = false;  // the truth view shows the right image
        }
        const cv::Size from = fromLeft ? sl : sr, to = fromLeft ? sr : sl;
        if (q.x < 0 || q.y < 0 || q.x >= from.width || q.y >= from.height) return {};

        const cv::Vec3d line = fromLeft ? g::lineInImage2(F, q) : g::lineInImage1(F, q);
        const Pen pen{colors::accent, 2.0};
        const int target = fromLeft ? kRightView : kLeftView, source = fromLeft ? kLeftView : kRightView;
        std::vector<ClickShapes> s;
        s.push_back({source, {PointsShape{QPolygonF{qp(q)}, colors::accent, 5.0}}});
        ClickShapes t{target, {}};
        if (auto l = lineShape(line, to, {0, 0}, pen)) t.shapes.push_back(*l);
        else t.shapes.push_back(LabelShape{QPointF(12, 12), QStringLiteral("the epipolar line misses this image"), colors::accent});
        s.push_back(t);
        ClickShapes m{kMatchesView, {}};
        const cv::Point2d srcOff = fromLeft ? cv::Point2d(0, 0) : cv::Point2d(offR, 0);
        const cv::Point2d dstOff = fromLeft ? cv::Point2d(offR, 0) : cv::Point2d(0, 0);
        m.shapes.push_back(PointsShape{QPolygonF{qp(q + srcOff)}, colors::accent, 5.0});
        if (auto l = lineShape(line, to, dstOff, pen)) m.shapes.push_back(*l);
        s.push_back(m);
        if (views.size() > 3 && views[3].name == QStringLiteral("Truth") && fromLeft) {  // the right image again
            ClickShapes tv{3, {}};
            if (auto l = lineShape(line, to, {0, 0}, pen)) tv.shapes.push_back(*l);
            s.push_back(tv);
        }
        return s;
    }

private:
    mutable FeatureCache cache_;
};

}  // namespace

VL_REGISTER_MODULE(EpipolarModule);

}  // namespace vl::app
