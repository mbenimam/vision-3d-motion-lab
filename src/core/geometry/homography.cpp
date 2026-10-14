// Homography by the normalised DLT inside RANSAC. Course "3D Vision and Motion", parts 2 and 4.
// The steps are listed in homography.hpp. Cost: one 2n × 9 SVD per fit, O(n) per residual pass.

#include "core/geometry/homography.hpp"

#include <algorithm>
#include <cmath>

namespace vl::geometry {

cv::Matx33d hartleyNormalisation(const Points& pts) {
    const double n = double(std::max<size_t>(pts.size(), 1));
    cv::Point2d centroid(0, 0);
    for (const auto& p : pts) centroid += p;
    centroid *= 1.0 / n;
    double meanDistance = 0;
    for (const auto& p : pts) meanDistance += std::hypot(p.x - centroid.x, p.y - centroid.y);
    meanDistance /= n;
    const double s = meanDistance > 1e-12 ? std::sqrt(2.0) / meanDistance : 1.0;
    return {s, 0, -s * centroid.x, 0, s, -s * centroid.y, 0, 0, 1};
}

namespace {

Points applyToAll(const cv::Matx33d& T, const Points& pts) {
    Points out;
    for (const auto& p : pts) out.push_back(transform(T, p));
    return out;
}

// The 9 entries of the null vector of A, read as a 3×3 matrix row by row.
cv::Matx33d nullVectorAsMatrix(const cv::Mat& vt) {
    cv::Matx33d M;
    for (int k = 0; k < 9; ++k) M(k / 3, k % 3) = vt.at<double>(8, k);
    return M;
}

}  // namespace

std::optional<cv::Matx33d> dltHomography(const Points& x, const Points& xp, bool normalise) {
    const int n = int(x.size());
    if (n < 4 || xp.size() != x.size()) return std::nullopt;
    const cv::Matx33d T = normalise ? hartleyNormalisation(x) : cv::Matx33d::eye();
    const cv::Matx33d Tp = normalise ? hartleyNormalisation(xp) : cv::Matx33d::eye();
    const Points a = applyToAll(T, x), b = applyToAll(Tp, xp);

    // x' × (H x) = 0 with x = (x, y, 1), x' = (u, v, 1) gives the two rows
    //   [ 0 0 0  −x −y −1   v x  v y  v ]
    //   [ x y 1   0  0  0  −u x −u y −u ]
    // With 4 points a zero 9th row makes A square, so its full SVD has the null vector in Vᵀ's last row.
    cv::Mat A = cv::Mat::zeros(std::max(2 * n, 9), 9, CV_64F);
    for (int i = 0; i < n; ++i) {
        const double X = a[size_t(i)].x, Y = a[size_t(i)].y, u = b[size_t(i)].x, v = b[size_t(i)].y;
        const double row0[9] = {0, 0, 0, -X, -Y, -1, v * X, v * Y, v};
        const double row1[9] = {X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u};
        for (int k = 0; k < 9; ++k) {
            A.at<double>(2 * i, k) = row0[k];
            A.at<double>(2 * i + 1, k) = row1[k];
        }
    }
    cv::Mat w, U, vt;
    cv::SVD::compute(A, w, U, vt);
    // Two vanishing singular values: the points do not pin H down (collinear, repeated...).
    if (w.at<double>(7) < 1e-12 * std::max(1.0, w.at<double>(0))) return std::nullopt;
    // H̃ maps normalised points: T' x' = H̃ T x, so H = T'⁻¹ H̃ T.
    cv::Matx33d H = Tp.inv() * nullVectorAsMatrix(vt) * T;
    if (std::abs(cv::determinant(H)) < 1e-300) return std::nullopt;
    if (std::abs(H(2, 2)) > 1e-12) H *= 1.0 / H(2, 2);
    return H;
}

void transferErrors(const cv::Matx33d& H, const Points& x, const Points& xp, std::vector<double>& out) {
    out.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const cv::Point2d q = transform(H, x[i]);
        out[i] = std::hypot(q.x - xp[i].x, q.y - xp[i].y);
    }
}

double cornerError(const cv::Matx33d& H, const cv::Matx33d& truth, cv::Size size) {
    const double W = size.width, Hh = size.height;
    const cv::Point2d corners[] = {{0, 0}, {W, 0}, {W, Hh}, {0, Hh}};
    double e = 0;
    for (const auto& c : corners) {
        const cv::Point2d a = transform(H, c), b = transform(truth, c);
        e += std::hypot(a.x - b.x, a.y - b.y);
    }
    return e / 4.0;
}

bool collinearTriple(const cv::Point2d* p) {
    // Twice the triangle's area relative to its squared longest side: about 0 means collinear.
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            for (int k = j + 1; k < 4; ++k) {
                const cv::Point2d u = p[j] - p[i], v = p[k] - p[i], w = p[k] - p[j];
                const double area = std::abs(u.x * v.y - u.y * v.x);
                const double longest = std::max({u.dot(u), v.dot(v), w.dot(w)});
                if (area <= 1e-3 * longest) return true;
            }
    return false;
}

namespace {

// What RANSAC needs to fit homographies (see ransac.hpp).
struct HomographyProblem {
    using Model = cv::Matx33d;
    const Points& x;
    const Points& xp;
    bool normalise;

    int size() const { return int(x.size()); }
    int sampleSize() const { return 4; }

    std::vector<Model> fitMinimal(const std::vector<int>& sample) const {
        cv::Point2d a[4], b[4];
        for (int k = 0; k < 4; ++k) {
            a[k] = x[size_t(sample[size_t(k)])];
            b[k] = xp[size_t(sample[size_t(k)])];
        }
        if (collinearTriple(a) || collinearTriple(b)) return {};
        const auto H = dltHomography(Points(a, a + 4), Points(b, b + 4), normalise);
        if (!H) return {};
        return {*H};
    }

    void residuals(const Model& H, std::vector<double>& r) const { transferErrors(H, x, xp, r); }

    std::optional<Model> fitAll(const std::vector<int>& indices) const {
        Points a, b;
        for (int i : indices) {
            a.push_back(x[size_t(i)]);
            b.push_back(xp[size_t(i)]);
        }
        return dltHomography(a, b, normalise);
    }
};

}  // namespace

HomographyEstimate estimateHomography(const Points& x, const Points& xp, const RansacParams& p, bool normalise,
                                      std::stop_token stop) {
    const auto r = ransac(HomographyProblem{x, xp, normalise}, p, stop);
    HomographyEstimate e;
    e.report = r.report;
    e.inliers = r.inliers;
    e.nInliers = r.nInliers;
    if (!r.model) return e;
    e.H = *r.model;
    e.ok = true;
    // RMS of the transfer error over the inliers.
    std::vector<double> err;
    transferErrors(e.H, x, xp, err);
    double sum = 0;
    for (size_t i = 0; i < err.size(); ++i)
        if (e.inliers[i]) sum += err[i] * err[i];
    e.rms = e.nInliers ? std::sqrt(sum / e.nInliers) : 0.0;
    return e;
}

}  // namespace vl::geometry
