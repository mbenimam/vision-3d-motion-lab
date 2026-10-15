#include "app/modules/scanner/scanner_samples.hpp"

#include "core/scanner/synthetic_page.hpp"

namespace vl::app::scanner_samples {
namespace {

Sample synthetic(const QString& label, vl::scanner::DeskScene scene) {
    return {label, [label, scene] {
                const auto d = vl::scanner::documentPhoto(scene);
                cv::Mat corners(4, 2, CV_64F);
                for (int i = 0; i < 4; ++i) {
                    corners.at<double>(i, 0) = d.corners[size_t(i)].x;
                    corners.at<double>(i, 1) = d.corners[size_t(i)].y;
                }
                Inputs in;
                in.name = label;
                in.images = {d.photo};
                in.truth = {corners, d.page};
                return in;
            }};
}

}  // namespace

std::vector<Sample> all() {
    using vl::scanner::DeskScene;
    return {synthetic(QStringLiteral("Synthetic: easy"), DeskScene::Easy),
            synthetic(QStringLiteral("Synthetic: strong perspective"), DeskScene::StrongPerspective),
            synthetic(QStringLiteral("Synthetic: hand shadow"), DeskScene::Shadowed),
            synthetic(QStringLiteral("Synthetic: low contrast"), DeskScene::LowContrast),
            synthetic(QStringLiteral("Synthetic: cluttered, corner hidden"), DeskScene::Cluttered)};
}

std::optional<vl::scanner::Quad> trueCorners(const Inputs& in) {
    if (in.truth.empty() || in.truth[0].rows != 4 || in.truth[0].type() != CV_64F) return std::nullopt;
    vl::scanner::Quad q;
    for (int i = 0; i < 4; ++i) q[size_t(i)] = {in.truth[0].at<double>(i, 0), in.truth[0].at<double>(i, 1)};
    return q;
}

cv::Mat truePage(const Inputs& in) { return in.truth.size() > 1 ? in.truth[1] : cv::Mat(); }

}  // namespace vl::app::scanner_samples
