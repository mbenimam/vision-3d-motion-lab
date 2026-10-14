#include "core/geometry/fundamental.hpp"

#include <cmath>
#include <limits>
#include <span>

namespace vl::geometry {

std::optional<cv::Matx33d> eightPoint(const Points& x1, const Points& x2, EightPointOptions opt) {
    const int n = static_cast<int>(x1.size());
    if (n < 8 || x2.size() != x1.size()) return std::nullopt;
    const cv::Matx33d T1 = opt.normalise ? hartleyNormalisation(x1) : cv::Matx33d::eye();
    const cv::Matx33d T2 = opt.normalise ? hartleyNormalisation(x2) : cv::Matx33d::eye();

    // One row per pair: x2ᵀ F x1 = 0 expanded over f = (F11 F12 F13 F21 ... F33). A 9th zero row for
    // exactly 8 points makes the system square (null vector = last row of Vᵀ).
    cv::Mat A = cv::Mat::zeros(std::max(n, 9), 9, CV_64F);
    for (int i = 0; i < n; ++i) {
        const cv::Point2d a = opt.normalise ? transform(T1, x1[size_t(i)]) : x1[size_t(i)];
        const cv::Point2d b = opt.normalise ? transform(T2, x2[size_t(i)]) : x2[size_t(i)];
        double* r = A.ptr<double>(i);
        r[0] = b.x * a.x, r[1] = b.x * a.y, r[2] = b.x;
        r[3] = b.y * a.x, r[4] = b.y * a.y, r[5] = b.y;
        r[6] = a.x, r[7] = a.y, r[8] = 1;
    }
    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A);  // thin: A has at least 9 rows
    if (w.at<double>(7) < 1e-12 * std::max(1.0, w.at<double>(0))) return std::nullopt;  // f not unique
    cv::Matx33d F;
    for (int k = 0; k < 9; ++k) F(k / 3, k % 3) = vt.at<double>(8, k);

    if (opt.enforceRank2) {
        cv::Matx31d s;
        cv::Matx33d U, Vt;
        cv::SVD::compute(F, s, U, Vt);
        F = U * cv::Matx33d::diag({s(0), s(1), 0.0}) * Vt;
    }
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

FundamentalEstimate estimateFundamental(const Points& x1, const Points& x2, const RansacParams& p,
                                        EightPointOptions opt, std::stop_token stop) {
    const int n = static_cast<int>(x1.size());
    auto gather = [&](std::span<const int> idx, Points& a, Points& b) {
        a.resize(idx.size());
        b.resize(idx.size());
        for (size_t k = 0; k < idx.size(); ++k) a[k] = x1[size_t(idx[k])], b[k] = x2[size_t(idx[k])];
    };
    RansacProblem<cv::Matx33d> pb;
    pb.size = n;
    pb.sampleSize = 8;
    pb.fit = [&](std::span<const int> idx) -> std::vector<cv::Matx33d> {
        Points a, b;
        gather(idx, a, b);
        const auto F = eightPoint(a, b, opt);
        return F ? std::vector<cv::Matx33d>{*F} : std::vector<cv::Matx33d>{};
    };
    pb.residuals = [&](const cv::Matx33d& F, std::vector<double>& r) {
        r.resize(size_t(n));
        for (int i = 0; i < n; ++i) r[size_t(i)] = sampsonDistance(F, x1[size_t(i)], x2[size_t(i)]);
    };
    pb.refit = [&](std::span<const int> idx) {
        Points a, b;
        gather(idx, a, b);
        return eightPoint(a, b, opt);
    };
    const auto r = ransac(pb, p, stop);

    FundamentalEstimate e;
    e.report = r.report;
    e.inliers = r.inliers;
    e.nInliers = r.nInliers;
    if (!r.model) return e;
    e.F = *r.model;
    e.ok = true;
    double s = 0;
    for (int i = 0; i < n; ++i)
        if (e.inliers[size_t(i)]) s += sampsonDistance(e.F, x1[size_t(i)], x2[size_t(i)]);
    e.meanSampson = e.nInliers ? s / e.nInliers : 0.0;
    cv::Matx31d sv;
    cv::SVD::compute(e.F, sv);
    e.singularValues = {sv(0), sv(1), sv(2)};
    return e;
}

}  // namespace vl::geometry
