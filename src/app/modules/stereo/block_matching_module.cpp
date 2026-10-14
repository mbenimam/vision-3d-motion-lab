// Topic: local stereo by block matching (SAD, SSD, ZNCC), with the left-right consistency check.

#include <cmath>

#include <opencv2/imgproc.hpp>

#include "app/module.hpp"
#include "app/modules/stereo/stereo_samples.hpp"
#include "core/common/image.hpp"
#include "core/stereo/block_matching.hpp"

namespace vl::app {
namespace {

class BlockMatchingModule final : public Module {
public:
    Info info() const override {
        return {QStringLiteral("stereo-local"), QStringLiteral("Block matching"), QStringLiteral("Stereo"),
                QStringLiteral("Two cameras side by side see a point at slightly different columns; that shift, the "
                               "<i>disparity</i>, is inversely proportional to depth. For each left pixel we slide a "
                               "small window along the same row of the right image and keep the shift where the two "
                               "windows look most alike."),
                QStringLiteral("switch the cost from SAD to ZNCC on <i>Cones</i>, then grow the window and watch thin "
                               "objects swell while noise disappears."),
                InputKind::ImagePair, 300, QStringLiteral("Disparity")};
    }

    std::vector<ParamSpec> params() const override {
        return {
            ParamSpec::choice(QStringLiteral("cost"), QStringLiteral("Similarity"), {"SAD", "SSD", "ZNCC"}, 2)
                .withHelp(QStringLiteral("SAD and SSD compare raw intensities; ZNCC removes each window's mean and "
                                         "contrast, so it survives a brightness change between the cameras.")),
            ParamSpec::integer(QStringLiteral("radius"), QStringLiteral("Window radius"), 1, 12, 3, 1, QStringLiteral(" px"))
                .withHelp(QStringLiteral("The window is (2r+1)² pixels. Larger: smoother and more reliable, but "
                                         "depth edges spread.")),
            ParamSpec::integer(QStringLiteral("maxd"), QStringLiteral("Largest disparity searched"), 8, 96, 64, 4, QStringLiteral(" px")),
            ParamSpec::toggle(QStringLiteral("lr"), QStringLiteral("Left-right consistency check"), true)
                .withHelp(QStringLiteral("Match right-to-left too and reject pixels where both directions disagree: "
                                         "mostly occlusions and textureless areas.")),
            ParamSpec::toggle(QStringLiteral("subpixel"), QStringLiteral("Sub-pixel refinement"), true),
        };
    }

    std::vector<Sample> samples() const override { return stereo_samples::all(); }

    Output compute(const Inputs& in, const Params& p, const RunContext& ctx) const override {
        stereo::BlockMatchParams bp;
        bp.cost = static_cast<stereo::Cost>(intParam(p, "cost"));
        bp.radius = intParam(p, "radius");
        bp.maxDisparity = intParam(p, "maxd");
        bp.leftRightCheck = toggleParam(p, "lr");
        bp.subpixel = toggleParam(p, "subpixel");
        const auto res = stereo::blockMatch(in.images[0], in.images[1], bp, ctx.stop, ctx.progress);

        const bool hasTruth = !in.truth.empty() && !in.truth[0].empty();
        const cv::Mat truth = hasTruth ? in.truth[0] : cv::Mat();
        const cv::Mat visible = in.truth.size() > 1 ? in.truth[1] : cv::Mat();
        const double hi = bp.maxDisparity;
        const cv::Mat valid = res.disparity == res.disparity;  // false on NaN
        cv::Mat dispShown;
        res.disparity.copyTo(dispShown);
        dispShown.setTo(0, ~valid);

        Frame f;
        View left{QStringLiteral("Left"), toQImage(in.images[0])};
        View right{QStringLiteral("Right"), toQImage(in.images[1])};
        const cv::Mat d = res.disparity;
        View disp{QStringLiteral("Disparity"), toQImage(vl::colorize(dispShown, 0, hi, valid)), {}, {}, d,
                  Legend{QStringLiteral("disparity (px) · black: rejected"), 0, hi, {}, {}}};
        disp.probe = [d, truth](QPoint q) {
            const float v = d.at<float>(q.y(), q.x());
            QString s = std::isfinite(v) ? QStringLiteral("d = %1 px").arg(v, 0, 'f', 1) : QStringLiteral("rejected");
            if (!truth.empty() && truth.at<float>(q.y(), q.x()) > 0)
                s += QStringLiteral("  ·  truth %1 px").arg(truth.at<float>(q.y(), q.x()), 0, 'f', 1);
            return s;
        };
        left.probe = disp.probe;
        f.views = {left, right, disp};

        Output out;
        if (hasTruth) {
            const double bad = stereo::badPixelRate(res.disparity, truth, visible, 1.0);
            out.stats.push_back({QStringLiteral("Bad pixels (error > 1 px)"), fmt(100 * bad) + QStringLiteral(" %"), true});
            // Error picture: the left image greyed out, wrong pixels in pink, rejected ones in blue.
            cv::Mat g;
            cv::cvtColor(vl::toBGR(in.images[0]), g, cv::COLOR_BGR2GRAY);
            cv::Mat err;
            cv::cvtColor(g / 2, err, cv::COLOR_GRAY2BGR);
            double sumAbs = 0;
            long n = 0;
            for (int y = 0; y < d.rows; ++y)
                for (int x = 0; x < d.cols; ++x) {
                    const float t = truth.at<float>(y, x);
                    if (t <= 0 || (!visible.empty() && !visible.at<uchar>(y, x))) continue;
                    const float v = d.at<float>(y, x);
                    if (!std::isfinite(v)) {
                        err.at<cv::Vec3b>(y, x) = {247, 195, 79};
                    } else if (std::abs(v - t) > 1.0f) {
                        err.at<cv::Vec3b>(y, x) = {138, 92, 255};
                    }
                    if (std::isfinite(v)) {
                        sumAbs += std::abs(v - t);
                        ++n;
                    }
                }
            out.stats.push_back({QStringLiteral("Mean |error| (kept pixels)"), fmt(n ? sumAbs / double(n) : 0, 2) + QStringLiteral(" px")});
            View e{QStringLiteral("Errors"), toQImage(err), {}, disp.probe, d};
            f.views.push_back(e);
            double tmax = 0;
            cv::minMaxLoc(truth, nullptr, &tmax);
            if (tmax > bp.maxDisparity)
                out.note = QStringLiteral("The true disparity reaches %1 px, beyond the search range: raise the "
                                          "largest disparity.").arg(tmax, 0, 'f', 0);
        }
        const double rejected = 100.0 * cv::countNonZero(res.rejected) / double(res.rejected.total());
        out.stats.push_back({QStringLiteral("Rejected by the L-R check"), fmt(rejected) + QStringLiteral(" %")});
        out.stats.push_back({QStringLiteral("Cost evaluations"),
                             QString::number(double(d.total()) * (bp.maxDisparity + 1), 'g', 3)});
        out.frames.push_back(std::move(f));
        return out;
    }

    // Click a left pixel: show its window, the row it is searched along in the right image, and the match.
    std::vector<ClickShapes> onClick(const Output& out, int, int view, QPointF px) const override {
        if (out.frames.empty() || view == 1) return {};
        const auto& f = out.frames.front();
        const cv::Mat& disp = f.views[2].data;
        const int x = int(px.x()), y = int(px.y());
        if (x < 0 || y < 0 || x >= disp.cols || y >= disp.rows) return {};
        const double d = disp.at<float>(y, x);
        const bool ok = std::isfinite(d);
        Pen row{colors::info, 1.0, true};
        std::vector<ClickShapes> s(4);
        for (int v = 0; v < 4; ++v) s[size_t(v)].view = v;
        const double r = 4;
        for (int v : {0, 2, 3}) {
            s[size_t(v)].shapes.push_back(RectShape{QRectF(x - r, y - r, 2 * r + 1, 2 * r + 1), {}});
        }
        s[1].shapes.push_back(LineShape{QLineF(0, y + 0.5, disp.cols, y + 0.5), row});
        s[1].shapes.push_back(LabelShape{QPointF(4, y - 18), QStringLiteral("search row y = %1").arg(y), colors::info});
        if (ok) {
            s[1].shapes.push_back(RectShape{QRectF(x - d - r, y - r, 2 * r + 1, 2 * r + 1), {}});
            s[1].shapes.push_back(LineShape{QLineF(x + 0.5, y - 10, x + 0.5, y + 10), {colors::muted, 1.0, true}});
            s[1].shapes.push_back(LabelShape{QPointF(x - d + r + 4, y + 6), QStringLiteral("match: %1 px to the left").arg(d, 0, 'f', 1)});
        }
        return s;
    }
};

}  // namespace

VL_REGISTER_MODULE(BlockMatchingModule);

}  // namespace vl::app
