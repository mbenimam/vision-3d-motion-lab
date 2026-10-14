#include "core/geometry/homography.hpp"

#include <cmath>
#include <span>

namespace vl::geometry {

cv::Matx33d hartleyNormalisation(const Points& pts) {
    cv::Point2d c(0, 0);
    for (const auto& p : pts) c += p;
    c *= 1.0 / double(std::max<size_t>(pts.size(), 1));
    double d = 0;
    for (const auto& p : pts) d += std::hypot(p.x - c.x, p.y - c.y);
    d /= double(std::max<size_t>(pts.size(), 1));
    const double s = d > 1e-12 ? std::sqrt(2.0) / d : 1.0;
    return {s, 0, -s * c.x, 0, s, -s * c.y, 0, 0, 1};
}

namespace {

Points apply(const cv::Matx33d& T, const Points& pts) {
    Points out(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) out[i] = transform(T, pts[i]);
    return out;
}

}  // namespace

std::optional<cv::Matx33d> dltHomography(const Points& src, const Points& dst, bool normalise) {
    const int n = static_cast<int>(src.size());
    if (n < 4 || dst.size() != src.size()) return std::nullopt;
    const cv::Matx33d T1 = normalise ? hartleyNormalisation(src) : cv::Matx33d::eye();
    const cv::Matx33d T2 = normalise ? hartleyNormalisation(dst) : cv::Matx33d::eye();
    const Points a = normalise ? apply(T1, src) : src;
    const Points b = normalise ? apply(T2, dst) : dst;

    // Rows of x' × (H x) = 0 for x = (x, y, 1), x' = (u, v, 1):
    //   [ 0 0 0  -x -y -1   v x  v y  v ]
    //   [ x y 1   0  0  0  -u x -u y -u ]
    // A square 9x9 system (an extra zero row for 4 points) keeps cv::SVD on its fast path.
    cv::Mat A = cv::Mat::zeros(std::max(2 * n, 9), 9, CV_64F);
    for (int i = 0; i < n; ++i) {
        const double x = a[size_t(i)].x, y = a[size_t(i)].y, u = b[size_t(i)].x, v = b[size_t(i)].y;
        double* r0 = A.ptr<double>(2 * i);
        double* r1 = A.ptr<double>(2 * i + 1);
        r0[3] = -x, r0[4] = -y, r0[5] = -1, r0[6] = v * x, r0[7] = v * y, r0[8] = v;
        r1[0] = x, r1[1] = y, r1[2] = 1, r1[6] = -u * x, r1[7] = -u * y, r1[8] = -u;
    }
    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A);  // thin: A has at least 9 rows
    // Two or more vanishing singular values: the points do not pin H down (collinear, repeated...).
    if (w.at<double>(7) < 1e-12 * std::max(1.0, w.at<double>(0))) return std::nullopt;
    cv::Matx33d Hn;
    for (int k = 0; k < 9; ++k) Hn(k / 3, k % 3) = vt.at<double>(8, k);
    cv::Matx33d H = T2.inv() * Hn * T1;
    if (std::abs(cv::determinant(H)) < 1e-300) return std::nullopt;
    if (std::abs(H(2, 2)) > 1e-12) H *= 1.0 / H(2, 2);
    return H;
}

void transferErrors(const cv::Matx33d& H, const Points& src, const Points& dst, std::vector<double>& out) {
    out.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        const cv::Point2d q = transform(H, src[i]);
        out[i] = std::hypot(q.x - dst[i].x, q.y - dst[i].y);
    }
}

double cornerError(const cv::Matx33d& H, const cv::Matx33d& truth, cv::Size size) {
    const cv::Point2d corners[] = {{0, 0}, {double(size.width), 0}, {double(size.width), double(size.height)},
                                   {0, double(size.height)}};
    double e = 0;
    for (const auto& c : corners) {
        const cv::Point2d a = transform(H, c), b = transform(truth, c);
        e += std::hypot(a.x - b.x, a.y - b.y);
    }
    return e / 4.0;
}

bool collinearTriple(const cv::Point2d* p) {
    // Twice the triangle area relative to the squared longest side: ~0 means collinear.
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            for (int k = j + 1; k < 4; ++k) {
                const cv::Point2d u = p[j] - p[i], v = p[k] - p[i];
                const double area = std::abs(u.x * v.y - u.y * v.x);
                const double scale = std::max({u.dot(u), v.dot(v), (p[k] - p[j]).dot(p[k] - p[j])});
                if (area <= 1e-3 * scale) return true;
            }
    return false;
}

HomographyEstimate estimateHomography(const Points& src, const Points& dst, const RansacParams& p, bool normalise,
                                      std::stop_token stop) {
    RansacProblem<cv::Matx33d> pb;
    pb.size = static_cast<int>(src.size());
    pb.sampleSize = 4;
    pb.fit = [&](std::span<const int> idx) -> std::vector<cv::Matx33d> {
        cv::Point2d a[4], b[4];
        for (int k = 0; k < 4; ++k) a[k] = src[size_t(idx[size_t(k)])], b[k] = dst[size_t(idx[size_t(k)])];
        if (collinearTriple(a) || collinearTriple(b)) return {};
        const auto H = dltHomography(Points(a, a + 4), Points(b, b + 4), normalise);
        return H ? std::vector<cv::Matx33d>{*H} : std::vector<cv::Matx33d>{};
    };
    pb.residuals = [&](const cv::Matx33d& H, std::vector<double>& r) { transferErrors(H, src, dst, r); };
    pb.refit = [&](std::span<const int> idx) {
        Points a, b;
        a.reserve(idx.size());
        b.reserve(idx.size());
        for (int i : idx) a.push_back(src[size_t(i)]), b.push_back(dst[size_t(i)]);
        return dltHomography(a, b, normalise);
    };
    const auto r = ransac(pb, p, stop);

    HomographyEstimate e;
    e.report = r.report;
    e.inliers = r.inliers;
    e.nInliers = r.nInliers;
    if (!r.model) return e;
    e.H = *r.model;
    e.ok = true;
    std::vector<double> err;
    transferErrors(e.H, src, dst, err);
    double s = 0;
    for (size_t i = 0; i < err.size(); ++i)
        if (e.inliers[i]) s += err[i] * err[i];
    e.rms = e.nInliers ? std::sqrt(s / e.nInliers) : 0.0;
    return e;
}

}  // namespace vl::geometry
