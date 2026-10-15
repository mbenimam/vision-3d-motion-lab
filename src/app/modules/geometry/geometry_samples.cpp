#include "app/modules/geometry/geometry_samples.hpp"

#include <fstream>
#include <stdexcept>

#include "core/geometry/scenes.hpp"

namespace vl::app::geometry_samples {
namespace {

// The Oxford affine-region files store a 3x3 matrix as 9 numbers, row by row.
cv::Mat readMatrix(const QString& path) {
    std::ifstream f(path.toStdString());
    cv::Mat H(3, 3, CV_64F);
    for (int k = 0; k < 9; ++k)
        if (!(f >> H.at<double>(k / 3, k % 3))) throw std::runtime_error("cannot read " + path.toStdString());
    return H;
}

bool allExist(const QStringList& rel) {
    for (const auto& r : rel)
        if (dataFile(r).isEmpty()) return false;
    return true;
}

QStringList paths(const QStringList& rel) {
    QStringList out;
    for (const auto& r : rel) out << dataFile(r);
    return out;
}

}  // namespace

std::vector<Sample> panorama() {
    std::vector<Sample> out;
    out.push_back({QStringLiteral("Synthetic: turning camera"), [] {
                       const auto v = vl::geometry::scenes::rotatingCamera({560, 420}, 3, 24.0, 21);
                       Inputs in;
                       in.name = QStringLiteral("synthetic");
                       in.images = v.images;
                       for (const auto& H : v.toReference) in.truth.push_back(cv::Mat(H).clone());
                       return in;
                   }});
    const QStringList graf = {"homography/graf/img1.ppm", "homography/graf/img3.ppm", "homography/graf/H1to3p"};
    if (allExist(graf))
        out.push_back({QStringLiteral("Graffiti 1 → 3 (true H)"), [graf] {
                           Inputs in;
                           in.name = QStringLiteral("graffiti 1-3");
                           in.images = loadImages(paths(graf.mid(0, 2)));
                           in.truth = {readMatrix(dataFile(graf[2])), cv::Mat::eye(3, 3, CV_64F)};
                           return in;
                       }});
    const QStringList pano = {"panorama/a1.png", "panorama/a2.png", "panorama/a3.png"};
    if (allExist(pano))
        out.push_back({QStringLiteral("Cathedral a1 · a2 · a3"), [pano] {
                           Inputs in;
                           in.name = QStringLiteral("a1-a3");
                           in.images = loadImages(paths(pano));
                           return in;
                       }});
    return out;
}

std::vector<Sample> epipolar() {
    std::vector<Sample> out;
    out.push_back({QStringLiteral("Synthetic: room (exact F)"), [] {
                       const auto s = vl::geometry::scenes::room({768, 576}, 3);
                       Inputs in;
                       in.name = QStringLiteral("synthetic room");
                       in.images = {s.left, s.right};
                       cv::Mat corr(int(s.x1.size()), 4, CV_64F);
                       for (int i = 0; i < corr.rows; ++i) {
                           double* r = corr.ptr<double>(i);
                           r[0] = s.x1[size_t(i)].x, r[1] = s.x1[size_t(i)].y, r[2] = s.x2[size_t(i)].x, r[3] = s.x2[size_t(i)].y;
                       }
                       in.truth = {cv::Mat(s.F).clone(), corr};
                       return in;
                   }});
    const std::pair<const char*, QStringList> sets[] = {
        {"Merton 001 / 002", {"epipolar/merton/001.jpg", "epipolar/merton/002.jpg"}},
        {"Merton 001 / 003", {"epipolar/merton/001.jpg", "epipolar/merton/003.jpg"}},
        {"Graffiti 1 / 2 (a plane)", {"homography/graf/img1.ppm", "homography/graf/img2.ppm"}},
    };
    for (const auto& [label, files] : sets) {
        if (!allExist(files)) continue;
        const QString name = QString::fromUtf8(label);
        out.push_back({name, [name, files] {
                           Inputs in;
                           in.name = name;
                           in.images = loadImages(paths(files));
                           return in;
                       }});
    }
    return out;
}

}  // namespace vl::app::geometry_samples
