// The fundamental matrix by the normalised 8-point algorithm inside RANSAC. Course "3D Vision and
// Motion", parts 4 and 5. The steps are listed in fundamental.hpp.

#include "core/geometry/fundamental.hpp"

#include <cmath>
#include <limits>

namespace vl::geometry {

namespace {

// Step 3: F = U diag(σ1, σ2, σ3) Vᵀ  ->  U diag(σ1, σ2, 0) Vᵀ.
cv::Matx33d nearestRankTwo(const cv::Matx33d& F) {
    cv::Matx31d sigma;
    cv::Matx33d U, Vt;
    cv::SVD::compute(F, sigma, U, Vt);
    return U * cv::Matx33d::diag({sigma(0), sigma(1), 0.0}) * Vt;
}

}  // namespace

std::optional<cv::Matx33d> eightPoint(const Points& x1, const Points& x2, EightPointOptions opt) {
    const int n = int(x1.size());
    if (n < 8 || x2.size() != x1.size()) return std::nullopt;
    // 1. Normalisation of each image's points.
    const cv::Matx33d T1 = opt.normalise ? hartleyNormalisation(x1) : cv::Matx33d::eye();
    const cv::Matx33d T2 = opt.normalise ? hartleyNormalisation(x2) : cv::Matx33d::eye();

    // 2. x2ᵀ F x1 = 0 with x1 = (x, y, 1), x2 = (x', y', 1), expanded over f = (F11 F12 F13 F21 ... F33):
    //      [ x'x  x'y  x'  y'x  y'y  y'  x  y  1 ] f = 0
    //    A 9th zero row for exactly 8 points makes A square (null vector = last row of Vᵀ).
    cv::Mat A = cv::Mat::zeros(std::max(n, 9), 9, CV_64F);
    for (int i = 0; i < n; ++i) {
        const cv::Point2d a = transform(T1, x1[size_t(i)]), b = transform(T2, x2[size_t(i)]);
        const double row[9] = {b.x * a.x, b.x * a.y, b.x, b.y * a.x, b.y * a.y, b.y, a.x, a.y, 1};
        for (int k = 0; k < 9; ++k) A.at<double>(i, k) = row[k];
    }
    cv::Mat w, U, vt;
    cv::SVD::compute(A, w, U, vt);
    if (w.at<double>(7) < 1e-12 * std::max(1.0, w.at<double>(0))) return std::nullopt;  // f not unique
    cv::Matx33d F;
    for (int k = 0; k < 9; ++k) F(k / 3, k % 3) = vt.at<double>(8, k);

    // 3. Rank 2, then back to pixels: x̃ = T x, so x2ᵀ T2ᵀ F̃ T1 x1 = 0 and F = T2ᵀ F̃ T1.
    if (opt.enforceRank2) F = nearestRankTwo(F);
    F = T2.t() * F * T1;
    const double norm = cv::norm(F);
    if (norm < 1e-300) return std::nullopt;
    return F * (1.0 / norm);
}

double sampsonDistance(const cv::Matx33d& F, const cv::Point2d& x1, const cv::Point2d& x2) {
    const cv::Vec3d a(x1.x, x1.y, 1), b(x2.x, x2.y, 1);
    const cv::Vec3d Fa = F * a, Ftb = F.t() * b;
    const double num = b.dot(Fa);
    const double den = Fa[0] * Fa[0] + Fa[1] * Fa[1] + Ftb[0] * Ftb[0] + Ftb[1] * Ftb[1];
    return den > 0 ? std::abs(num) / std::sqrt(den) : std::numeric_limits<double>::infinity();
}

double pointLineDistance(const cv::Vec3d& l, const cv::Point2d& p) {
    const double n = std::hypot(l[0], l[1]);
    return n > 0 ? std::abs(l[0] * p.x + l[1] * p.y + l[2]) / n : std::numeric_limits<double>::infinity();
}

double meanSymmetricEpipolarDistance(const cv::Matx33d& F, const Points& x1, const Points& x2) {
    if (x1.empty()) return 0;
    double s = 0;
    for (size_t i = 0; i < x1.size(); ++i)
        s += 0.5 * (pointLineDistance(lineInImage2(F, x1[i]), x2[i]) + pointLineDistance(lineInImage1(F, x2[i]), x1[i]));
    return s / double(x1.size());
}

Epipoles epipoles(const cv::Matx33d& F) {
    cv::Matx31d s;
    cv::Matx33d U, Vt;
    cv::SVD::compute(F, s, U, Vt, cv::SVD::FULL_UV);
    // F = U Σ Vᵀ: the right null vector (F e1 = 0) is the last row of Vᵀ, the left one the last column of U.
    return {cv::Vec3d(Vt(2, 0), Vt(2, 1), Vt(2, 2)), cv::Vec3d(U(0, 2), U(1, 2), U(2, 2))};
}

std::optional<cv::Point2d> finitePoint(const cv::Vec3d& h) {
    if (std::abs(h[2]) < 1e-9 * std::hypot(h[0], h[1])) return std::nullopt;
    return cv::Point2d(h[0] / h[2], h[1] / h[2]);
}

std::optional<std::pair<cv::Point2d, cv::Point2d>> clipLine(const cv::Vec3d& l, cv::Size size) {
    // Intersect with the four borders and keep the hits inside the rectangle (Liang–Barsky would do too).
    const double W = size.width, H = size.height, a = l[0], b = l[1], c = l[2];
    std::vector<cv::Point2d> hits;
    auto add = [&](double x, double y) {
        if (x < -1e-6 || x > W + 1e-6 || y < -1e-6 || y > H + 1e-6) return;
        for (const auto& q : hits)
            if (std::hypot(q.x - x, q.y - y) < 1e-6) return;
        hits.emplace_back(x, y);
    };
    if (std::abs(b) > 1e-12) {
        add(0, -c / b);
        add(W, -(c + a * W) / b);
    }
    if (std::abs(a) > 1e-12) {
        add(-c / a, 0);
        add(-(c + b * H) / a, H);
    }
    if (hits.size() < 2) return std::nullopt;
    return std::pair{hits[0], hits[1]};
}

cv::Matx33d fundamentalFromCameras(const cv::Matx33d& K1, const cv::Matx33d& K2, const cv::Matx33d& R,
                                   const cv::Vec3d& t) {
    const cv::Matx33d tx(0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0);
    cv::Matx33d F = K2.inv().t() * tx * R * K1.inv();
    return F * (1.0 / cv::norm(F));
}

namespace {

// What RANSAC needs to fit fundamental matrices (see ransac.hpp).
struct FundamentalProblem {
    using Model = cv::Matx33d;
    const Points& x1;
    const Points& x2;
    EightPointOptions options;

    int size() const { return int(x1.size()); }
    int sampleSize() const { return 8; }

    std::optional<Model> fitAll(const std::vector<int>& indices) const {
        Points a, b;
        for (int i : indices) {
            a.push_back(x1[size_t(i)]);
            b.push_back(x2[size_t(i)]);
        }
        return eightPoint(a, b, options);
    }

    std::vector<Model> fitMinimal(const std::vector<int>& sample) const {
        const auto F = fitAll(sample);
        if (!F) return {};
        return {*F};
    }

    void residuals(const Model& F, std::vector<double>& r) const {
        r.resize(x1.size());
        for (size_t i = 0; i < x1.size(); ++i) r[i] = sampsonDistance(F, x1[i], x2[i]);
    }
};

}  // namespace

FundamentalEstimate estimateFundamental(const Points& x1, const Points& x2, const RansacParams& p,
                                        EightPointOptions opt, std::stop_token stop) {
    const auto r = ransac(FundamentalProblem{x1, x2, opt}, p, stop);
    FundamentalEstimate e;
    e.report = r.report;
    e.inliers = r.inliers;
    e.nInliers = r.nInliers;
    if (!r.model) return e;
    e.F = *r.model;
    e.ok = true;
    double sum = 0;
    for (size_t i = 0; i < x1.size(); ++i)
        if (e.inliers[i]) sum += sampsonDistance(e.F, x1[i], x2[i]);
    e.meanSampson = e.nInliers ? sum / e.nInliers : 0.0;
    cv::Matx31d sigma;
    cv::SVD::compute(e.F, sigma);
    e.singularValues = {sigma(0), sigma(1), sigma(2)};
    return e;
}

}  // namespace vl::geometry
