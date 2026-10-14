// Topic: seed-and-grow stereo. Confident ZNCC matches spread to their neighbours, best first.

#include <cmath>

#include <opencv2/imgproc.hpp>

#include "app/module.hpp"
#include "app/modules/stereo/stereo_samples.hpp"
#include "app/modules/stereo/stereo_views.hpp"
#include "core/common/image.hpp"
#include "core/stereo/seeds.hpp"

namespace vl::app {
namespace {

namespace sv = stereo_views;

enum ViewIndex { kLeft, kRight, kSeeds, kGrowth, kDisparity, kErrors };

class SeedsModule final : public Module {
public:
    Info info() const override {
        return {QStringLiteral("stereo-seeds"), QStringLiteral("Seed propagation"), QStringLiteral("Stereo"),
                QStringLiteral("Trust only the matches you are sure of, then let them spread. Pixels whose window "
                               "correlates almost perfectly (ZNCC) and that pass the left-right check become "
                               "<i>seeds</i>; the best reached pixel is repeatedly taken from a priority queue and its "
                               "neighbours try only its disparity ± 1. Growth stops where no nearby disparity "
                               "correlates well: textureless areas, occlusions, depth edges."),
                QStringLiteral("on <i>Tsukuba</i>, raise the growth floor to 0.8 and watch the lamp and the "
                               "textureless wall stay unreached, then set seed spacing to 1 as in the course."),
                InputKind::ImagePair, 310, QStringLiteral("Growth")};
    }

    std::vector<ParamSpec> params() const override {
        return {
            ParamSpec::real(QStringLiteral("seed"), QStringLiteral("Seed threshold (ZNCC)"), 0.80, 0.995, 0.95, 0.005, 3)
                .withHelp(QStringLiteral("Correlation a pixel's best match needs to start a region.")),
            ParamSpec::integer(QStringLiteral("spacing"), QStringLiteral("Seed spacing"), 1, 32, 16, 1, QStringLiteral(" px"))
                .withHelp(QStringLiteral("Keep only the best seed in each cell of this size. 1 keeps every confident "
                                         "pixel; a few hundred seeds are enough to cover a textured pair.")),
            ParamSpec::real(QStringLiteral("stop"), QStringLiteral("Growth floor (ZNCC)"), 0.0, 0.95, 0.5, 0.05, 2)
                .withHelp(QStringLiteral("A neighbour is accepted only if its best of d-1, d, d+1 correlates at least "
                                         "this well.")),
            ParamSpec::integer(QStringLiteral("radius"), QStringLiteral("Window radius"), 1, 8, 3, 1, QStringLiteral(" px"))
                .withHelp(QStringLiteral("The ZNCC window is (2r+1)² pixels.")),
            ParamSpec::toggle(QStringLiteral("lr"), QStringLiteral("L-R check on seeds"), true)
                .withHelp(QStringLiteral("A seed must also be the best match when searching from the right image "
                                         "back to the left.")),
        };
    }

    std::vector<Sample> samples() const override { return stereo_samples::all(); }

    Output compute(const Inputs& in, const Params& p, const RunContext& ctx) const override {
        const sv::Working w = sv::prepare(in, 1);
        stereo::SeedParams sp;
        sp.maxDisparity = w.maxDisparity;
        sp.seedScore = realParam(p, "seed");
        sp.seedSpacing = intParam(p, "spacing");
        sp.stopScore = realParam(p, "stop");
        sp.radius = intParam(p, "radius");
        sp.leftRightCheck = toggleParam(p, "lr");

        // Partial pictures as the region grows; the seeds are the zeros of the order map.
        auto observer = [&](const cv::Mat& disparity, const cv::Mat& order, int reached) {
            if (!ctx.partial) return;
            auto out = std::make_shared<Output>();
            out->frames.push_back(frame(w, disparity, order, seedsFrom(order)));
            out->stats.push_back({QStringLiteral("Reached so far"),
                                  fmt(100.0 * reached / double(order.total())) + QStringLiteral(" %"), true});
            ctx.partial(out);
        };
        const int every = std::max(1, int(w.left.total() / 20));  // a picture every 5 % of the image
        const auto res = stereo::propagateSeeds(w.left, w.right, sp, ctx.stop, ctx.progress, observer, every);

        Output out;
        out.frames.push_back(frame(w, res.disparity, res.order, res.seeds));
        const cv::Mat reached = res.order >= 0;
        if (!w.truth.empty()) {
            out.stats.push_back({QStringLiteral("Bad pixels (non-occluded)"),
                                 fmt(100 * sv::badRate(res.disparity, w)) + QStringLiteral(" %"), true});
            sv::Working onlyReached = w;
            onlyReached.visible = w.visible & reached;
            out.stats.push_back({QStringLiteral("Errors among reached pixels"),
                                 fmt(100 * sv::badRate(res.disparity, onlyReached)) + QStringLiteral(" %")});
        }
        out.stats.push_back({QStringLiteral("Pixels reached"),
                             fmt(100.0 * cv::countNonZero(reached) / double(reached.total())) + QStringLiteral(" %")});
        out.stats.push_back({QStringLiteral("Seeds"), QString::number(res.seeds.size())});
        if (res.seeds.empty())
            out.note = QStringLiteral("No pixel reaches the seed threshold: lower it.");
        return out;
    }

    std::vector<ClickShapes> onClick(const Output& out, int, int view, QPointF px) const override {
        if (out.frames.empty() || view == kRight) return {};
        const auto& f = out.frames.front();
        return sv::matchClick(f.views[kDisparity].data, kRight, int(f.views.size()), px,
                              QStringLiteral("not reached: no match"));
    }

private:
    static std::vector<cv::Point> seedsFrom(const cv::Mat& order) {
        std::vector<cv::Point> s;
        for (int y = 0; y < order.rows; ++y)
            for (int x = 0; x < order.cols; ++x)
                if (order.at<int>(y, x) == 0) s.emplace_back(x, y);
        return s;
    }

    static Frame frame(const sv::Working& w, const cv::Mat& disparity, const cv::Mat& order,
                       const std::vector<cv::Point>& seeds) {
        Frame f;
        View disp = sv::disparityView(QStringLiteral("Disparity"), disparity, w, QStringLiteral("not reached"));
        View left{QStringLiteral("Left"), toQImage(w.left), {}, disp.probe};
        View right{QStringLiteral("Right"), toQImage(w.right)};

        // Seeds as small dots over the dimmed left image (painted: there can be thousands).
        cv::Mat seedImg = sv::dimmed(w.left);
        const int dot = seeds.size() > 5000 ? 0 : 1;
        for (const auto& s : seeds)
            cv::rectangle(seedImg, cv::Rect(s.x - dot, s.y - dot, 2 * dot + 1, 2 * dot + 1), cv::Scalar(0x00, 0xb0, 0xff), cv::FILLED);
        View seedView{QStringLiteral("Seeds"), toQImage(seedImg), {}, disp.probe};
        seedView.shapes.push_back(LabelShape{QPointF(6, 6), QStringLiteral("%1 seeds").arg(seeds.size())});

        // Order of arrival, normalised to [0, 1]: seeds at 0, the last pixel reached at 1.
        double last = 1;
        cv::minMaxLoc(order, nullptr, &last);
        cv::Mat t;
        order.convertTo(t, CV_32F, 1.0 / std::max(1.0, last));
        View growth{QStringLiteral("Growth"), toQImage(vl::colorize(t, 0, 1, order >= 0)), {}, {}, order,
                    Legend{QStringLiteral("order of arrival · black: not reached"), 0, 1, QStringLiteral("seeds"),
                           QStringLiteral("last")}};
        const cv::Mat o = order;
        const int n = int(last);
        growth.probe = [o, n](QPoint q) {
            if (q.x() < 0 || q.y() < 0 || q.x() >= o.cols || q.y() >= o.rows) return QString();
            const int k = o.at<int>(q.y(), q.x());
            if (k < 0) return QStringLiteral("not reached");
            if (k == 0) return QStringLiteral("seed");
            return QStringLiteral("reached %1 of %2").arg(k).arg(n);
        };

        f.views = {left, right, seedView, growth, disp};
        if (!w.truth.empty()) f.views.push_back(View{QStringLiteral("Errors"), toQImage(sv::errorImage(disparity, w)), {}, disp.probe, disparity});
        return f;
    }
};

}  // namespace

VL_REGISTER_MODULE(SeedsModule);

}  // namespace vl::app
