#include "core/geometry/scenes.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <random>

#include <opencv2/imgproc.hpp>

#include "core/common/synthetic.hpp"
#include "core/geometry/fundamental.hpp"

namespace vl::geometry::scenes {

cv::Matx33d rotation(double yawDeg, double pitchDeg, double rollDeg) {
    const double y = yawDeg * std::numbers::pi / 180, p = pitchDeg * std::numbers::pi / 180,
                 r = rollDeg * std::numbers::pi / 180;
    const cv::Matx33d Ry(std::cos(y), 0, std::sin(y), 0, 1, 0, -std::sin(y), 0, std::cos(y));
    const cv::Matx33d Rx(1, 0, 0, 0, std::cos(p), -std::sin(p), 0, std::sin(p), std::cos(p));
    const cv::Matx33d Rz(std::cos(r), -std::sin(r), 0, std::sin(r), std::cos(r), 0, 0, 0, 1);
    return Rz * Rx * Ry;
}

namespace {

// Scatter of soft-coloured shapes (discs, boxes, triangles, rings): corners and blobs for SIFT.
void scatterShapes(cv::Mat& img, std::mt19937& rng, int count, double scale) {
    std::uniform_real_distribution<double> ux(0, img.cols), uy(0, img.rows), u01(0, 1);
    for (int i = 0; i < count; ++i) {
        const cv::Point c(int(ux(rng)), int(uy(rng)));
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(u01(rng) * 180, 90 + 100 * u01(rng), 120 + 110 * u01(rng))), bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        const cv::Vec3b col = bgr.at<cv::Vec3b>(0);
        const cv::Scalar s(col[0], col[1], col[2]);
        const int r = int(6 + 40 * u01(rng) * u01(rng) * scale);
        switch (i % 4) {
            case 0: cv::circle(img, c, r, s, cv::FILLED, cv::LINE_AA); break;
            case 1: cv::rectangle(img, c, c + cv::Point(r * 2, int(r * (0.6 + u01(rng)))), s, cv::FILLED, cv::LINE_AA); break;
            case 2: {
                const std::vector<cv::Point> tri = {c, c + cv::Point(r * 2, int(r * 0.4)), c + cv::Point(int(r * 0.6), r * 2)};
                cv::fillConvexPoly(img, tri, s, cv::LINE_AA);
                break;
            }
            default: cv::circle(img, c, r, s, 2 + int(4 * u01(rng)), cv::LINE_AA);
        }
    }
}

// A colourful "poster wall": colour noise under a scatter of shapes and a few straight lines,
// so there are corners and blobs for SIFT and straight edges that show whether a warp is projective.
cv::Mat poster(cv::Size size, std::uint32_t seed) {
    cv::Mat base = vl::synth::colorTexture(size, seed);
    cv::multiply(base, cv::Scalar::all(0.55), base);
    cv::Mat img = base.clone();
    std::mt19937 rng(seed * 7919u + 13u);
    scatterShapes(img, rng, std::max(20, size.area() / 9000), size.width / 400.0 + 0.5);
    std::uniform_real_distribution<double> ux(0, size.width), uy(0, size.height);
    for (int i = 0; i < 8; ++i)
        cv::line(img, {int(ux(rng)), int(uy(rng))}, {int(ux(rng)), int(uy(rng))}, cv::Scalar::all(235), 2, cv::LINE_AA);
    cv::addWeighted(img, 0.72, base, 0.28, 0, img);  // shapes slightly see-through: softer, still sharp-edged
    return img;
}

// A calmer surface for the room: a flat paint colour mottled by smooth noise, optionally tiled
// (tile = side in pixels, 0 for none), with a sparse scatter of shapes.
cv::Mat surface(cv::Size size, std::uint32_t seed, cv::Scalar paint, int tile, int shapesPerMpx) {
    cv::Mat small = vl::synth::colorTexture(cv::Size(std::max(8, size.width / 6), std::max(8, size.height / 6)), seed);
    cv::Mat mottle;
    cv::resize(small, mottle, size, 0, 0, cv::INTER_CUBIC);  // the strokes become soft smudges
    cv::Mat img(size, CV_8UC3);
    for (int y = 0; y < size.height; ++y) {
        const cv::Vec3b* m = mottle.ptr<cv::Vec3b>(y);
        cv::Vec3b* o = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < size.width; ++x) {
            const double t = tile > 0 && ((x / tile + y / tile) % 2) ? 0.72 : 1.0;
            for (int k = 0; k < 3; ++k) o[x][k] = cv::saturate_cast<uchar>(t * (0.6 * paint[k] + 0.3 * m[x][k]));
        }
    }
    std::mt19937 rng(seed * 7919u + 13u);
    scatterShapes(img, rng, std::max(6, int(size.area() * 1e-6 * shapesPerMpx)), 1.2);
    return img;
}

}  // namespace

RotatingViews rotatingCamera(cv::Size viewSize, int nViews, double stepDeg, std::uint32_t seed) {
    const double f = 0.95 * viewSize.width;
    const cv::Matx33d K(f, 0, viewSize.width / 2.0, 0, f, viewSize.height / 2.0, 0, 0, 1);
    RotatingViews out;
    out.reference = nViews / 2;
    std::vector<cv::Matx33d> R;
    for (int i = 0; i < nViews; ++i)
        R.push_back(rotation((out.reference - i) * stepDeg, 2.0 * std::sin(i * 1.7), 1.5 * std::cos(i * 2.3)));

    // The backdrop is the plane z = 1 of the reference-free world frame, sampled with focal f: backdrop
    // pixel s <-> ray Ks⁻¹ s. Size it to the union of what the views see.
    const cv::Matx33d Kinv = K.inv();
    double x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (int i = 0; i < nViews; ++i)
        for (const cv::Point2d c : {cv::Point2d(0, 0), cv::Point2d(viewSize.width, 0),
                                    cv::Point2d(viewSize.width, viewSize.height), cv::Point2d(0, viewSize.height)}) {
            const cv::Vec3d ray = R[size_t(i)].t() * (Kinv * cv::Vec3d(c.x, c.y, 1));
            x0 = std::min(x0, f * ray[0] / ray[2]), x1 = std::max(x1, f * ray[0] / ray[2]);
            y0 = std::min(y0, f * ray[1] / ray[2]), y1 = std::max(y1, f * ray[1] / ray[2]);
        }
    const cv::Matx33d Ks(f, 0, 8 - x0, 0, f, 8 - y0, 0, 0, 1);
    const cv::Mat backdrop = poster(cv::Size(int(x1 - x0) + 16, int(y1 - y0) + 16), seed);

    std::mt19937 rng(seed);
    const double gains[] = {0.82, 1.0, 1.18, 0.9, 1.1};
    for (int i = 0; i < nViews; ++i) {
        // view(x) = backdrop(Ks Rᵢᵀ K⁻¹ x): exactly the projective map the camera applies.
        const cv::Matx33d M = Ks * R[size_t(i)].t() * Kinv;
        cv::Mat view;
        cv::warpPerspective(backdrop, view, cv::Mat(M), viewSize, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                            cv::BORDER_REPLICATE);
        cv::Mat noise(view.size(), CV_32FC3);
        cv::randn(noise, 0, 2.0);
        cv::Mat vf;
        view.convertTo(vf, CV_32FC3, gains[i % 5]);
        vf += noise;
        vf.convertTo(view, CV_8UC3);
        out.images.push_back(view);
        out.toReference.push_back(K * R[size_t(out.reference)] * R[size_t(i)].t() * Kinv);
    }
    for (auto& H : out.toReference) H *= 1.0 / H(2, 2);
    return out;
}

namespace {

// A textured rectangle X = o + a u + b v, a in [0, lu], b in [0, lv] (metres), texture at ppm px/m.
struct Quad {
    cv::Vec3d o, u, v;
    double lu, lv;
    cv::Mat tex;
    double shade;
};

constexpr double kPpm = 110.0;

struct Hit {
    double lambda = std::numeric_limits<double>::infinity();
    int quad = -1;
    double a = 0, b = 0;
};

// Nearest intersection of the ray C + λ d (λ > 0) with the quads.
Hit cast(const std::vector<Quad>& quads, const cv::Vec3d& C, const cv::Vec3d& d) {
    Hit h;
    for (int q = 0; q < static_cast<int>(quads.size()); ++q) {
        const Quad& Q = quads[size_t(q)];
        const cv::Vec3d n = Q.u.cross(Q.v);
        const double den = n.dot(d);
        if (std::abs(den) < 1e-12) continue;
        const double lambda = n.dot(Q.o - C) / den;
        if (lambda <= 1e-6 || lambda >= h.lambda) continue;
        const cv::Vec3d X = C + lambda * d - Q.o;
        const double a = X.dot(Q.u), b = X.dot(Q.v);
        if (a < 0 || b < 0 || a > Q.lu || b > Q.lv) continue;
        h = {lambda, q, a, b};
    }
    return h;
}

cv::Vec3b sample(const cv::Mat& tex, double x, double y) {
    const int x0 = std::clamp(int(x), 0, tex.cols - 2), y0 = std::clamp(int(y), 0, tex.rows - 2);
    const double fx = std::clamp(x - x0, 0.0, 1.0), fy = std::clamp(y - y0, 0.0, 1.0);
    const cv::Vec3b *r0 = tex.ptr<cv::Vec3b>(y0), *r1 = tex.ptr<cv::Vec3b>(y0 + 1);
    cv::Vec3b out;
    for (int k = 0; k < 3; ++k)
        out[k] = cv::saturate_cast<uchar>((1 - fy) * ((1 - fx) * r0[x0][k] + fx * r0[x0 + 1][k]) +
                                          fy * ((1 - fx) * r1[x0][k] + fx * r1[x0 + 1][k]));
    return out;
}

// Render camera K [R | t] by casting one ray per pixel, 2x2 supersampled.
cv::Mat render(const std::vector<Quad>& quads, const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t,
               cv::Size size) {
    const int ss = 2;
    const cv::Matx33d S(ss, 0, 0, 0, ss, 0, 0, 0, 1);
    const cv::Matx33d back = R.t() * (S * K).inv();  // pixel of the big image -> world ray
    const cv::Vec3d C = -(R.t() * t);
    cv::Mat big(size.height * ss, size.width * ss, CV_8UC3);
    cv::parallel_for_(cv::Range(0, big.rows), [&](const cv::Range& rows) {
        for (int y = rows.start; y < rows.end; ++y) {
            cv::Vec3b* o = big.ptr<cv::Vec3b>(y);
            for (int x = 0; x < big.cols; ++x) {
                const cv::Vec3d d = back * cv::Vec3d(x + 0.5 - 0.5 * ss, y + 0.5 - 0.5 * ss, 1);
                const Hit h = cast(quads, C, d);
                if (h.quad < 0) {
                    o[x] = {20, 20, 24};
                    continue;
                }
                const Quad& Q = quads[size_t(h.quad)];
                const cv::Vec3b c = sample(Q.tex, h.a * kPpm, h.b * kPpm);
                o[x] = {cv::saturate_cast<uchar>(c[0] * Q.shade), cv::saturate_cast<uchar>(c[1] * Q.shade),
                        cv::saturate_cast<uchar>(c[2] * Q.shade)};
            }
        }
    });
    cv::Mat out;
    cv::resize(big, out, size, 0, 0, cv::INTER_AREA);
    return out;
}

}  // namespace

TwoViewScene room(cv::Size size, std::uint32_t seed) {
    TwoViewScene s;
    const double f = 0.82 * size.width;
    s.K = cv::Matx33d(f, 0, size.width / 2.0, 0, f, size.height / 2.0, 0, 0, 1);
    // Camera 2 moves right, slightly up and mostly forward, and turns a little: the epipole (the image of
    // the other camera's centre) lands inside the first image, where all epipolar lines meet.
    const cv::Vec3d C2(0.55, -0.15, 1.3);
    s.R = rotation(-6.0, 2.5, 1.5);
    s.t = -(s.R * C2);
    s.F = fundamentalFromCameras(s.K, s.K, s.R, s.t);

    // paint is BGR; tile is the floor tile side in metres (0: plain).
    auto quad = [&](cv::Vec3d o, cv::Vec3d u, cv::Vec3d v, double lu, double lv, std::uint32_t sd, cv::Scalar paint,
                    double shade, double tile = 0, int shapes = 220) {
        const cv::Size ts(int(lu * kPpm) + 2, int(lv * kPpm) + 2);
        return Quad{o, u, v, lu, lv, surface(ts, seed + sd, paint, int(tile * kPpm), shapes), shade};
    };
    const cv::Vec3d X(1, 0, 0), Y(0, 1, 0), Z(0, 0, 1);
    std::vector<Quad> quads = {
        quad({-4, -3, 9}, X, Y, 8, 5, 1, {150, 160, 175}, 1.0),               // back wall, warm grey
        quad({-4, 2, 0}, X, Z, 8, 9, 2, {120, 150, 185}, 0.9, 0.5, 25),       // tiled floor (y down)
        quad({-4, -3, 0}, Z, Y, 9, 5, 3, {170, 140, 110}, 0.8),               // left wall, blue
        quad({4, -3, 0}, Z, Y, 9, 5, 4, {120, 165, 135}, 0.85),               // right wall, green
        quad({-4, -3, 0}, X, Z, 8, 9, 5, {150, 150, 150}, 0.55, 0, 10),       // ceiling
        quad({-1.4, 0.5, 4.6}, X, Y, 1.9, 1.5, 6, {70, 130, 200}, 1.05, 0, 120), // box: front, orange
        quad({0.5, 0.5, 4.6}, Z, Y, 1.6, 1.5, 7, {70, 130, 200}, 0.75, 0, 120),  // box: right side
        quad({-1.4, 0.5, 4.6}, X, Z, 1.9, 1.6, 8, {70, 130, 200}, 1.2, 0, 120),  // box: top
        quad({1.6, -1.8, 6.4}, X, Y, 1.1, 3.8, 9, {190, 120, 150}, 1.0, 0, 120), // a standing panel, violet
    };
    s.left = render(quads, s.K, cv::Matx33d::eye(), cv::Vec3d(0, 0, 0), size);
    s.right = render(quads, s.K, s.R, s.t, size);

    // True correspondences: random left pixels, their 3D point, kept if camera 2 sees the same point.
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ux(0, size.width), uy(0, size.height);
    const cv::Matx33d Kinv = s.K.inv();
    const cv::Vec3d C1(0, 0, 0);
    for (int tries = 0; tries < 4000 && s.x1.size() < 400; ++tries) {
        const cv::Point2d p(ux(rng), uy(rng));
        const cv::Vec3d d1 = Kinv * cv::Vec3d(p.x, p.y, 1);
        const Hit h1 = cast(quads, C1, d1);
        if (h1.quad < 0) continue;
        const cv::Vec3d P = C1 + h1.lambda * d1;
        const cv::Vec3d q = s.K * (s.R * P + s.t);
        if (q[2] <= 0) continue;
        const cv::Point2d p2(q[0] / q[2], q[1] / q[2]);
        if (p2.x < 0 || p2.y < 0 || p2.x >= size.width || p2.y >= size.height) continue;
        const Hit h2 = cast(quads, C2, s.R.t() * (Kinv * cv::Vec3d(p2.x, p2.y, 1)));
        if (h2.quad < 0 || cv::norm(C2 + h2.lambda * (s.R.t() * (Kinv * cv::Vec3d(p2.x, p2.y, 1))) - P) > 1e-6) continue;
        s.x1.push_back(p);
        s.x2.push_back(p2);
    }
    return s;
}

PointPairs homographyPairs(const cv::Matx33d& H, cv::Size size, int nInliers, int nOutliers, double noise,
                           std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ux(0, size.width), uy(0, size.height);
    std::normal_distribution<double> g(0, noise);
    PointPairs pp;
    while (int(pp.cleanA.size()) < nInliers) {
        const cv::Point2d a(ux(rng), uy(rng));
        const cv::Point2d b = transform(H, a);
        if (b.x < 0 || b.y < 0 || b.x >= size.width || b.y >= size.height) continue;
        pp.cleanA.push_back(a);
        pp.cleanB.push_back(b);
    }
    for (int i = 0; i < nInliers + nOutliers; ++i) {
        if (i < nInliers) {
            pp.a.push_back(pp.cleanA[size_t(i)] + cv::Point2d(g(rng), g(rng)));
            pp.b.push_back(pp.cleanB[size_t(i)] + cv::Point2d(g(rng), g(rng)));
            pp.isInlier.push_back(1);
        } else {
            pp.a.emplace_back(ux(rng), uy(rng));
            pp.b.emplace_back(ux(rng), uy(rng));
            pp.isInlier.push_back(0);
        }
    }
    // Shuffle so inliers are not all first (RANSAC must not depend on the order).
    std::vector<size_t> perm(pp.a.size());
    for (size_t i = 0; i < perm.size(); ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    PointPairs out = pp;
    for (size_t i = 0; i < perm.size(); ++i) {
        out.a[i] = pp.a[perm[i]];
        out.b[i] = pp.b[perm[i]];
        out.isInlier[i] = pp.isInlier[perm[i]];
    }
    return out;
}

PointPairs twoViewPairs(const cv::Matx33d& K, const cv::Matx33d& R, const cv::Vec3d& t, cv::Size size, int nInliers,
                        int nOutliers, double noise, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ux(0, size.width), uy(0, size.height), depth(3, 12);
    std::normal_distribution<double> g(0, noise);
    PointPairs pp;
    const cv::Matx33d Kinv = K.inv();
    while (int(pp.cleanA.size()) < nInliers) {
        const cv::Point2d a(ux(rng), uy(rng));
        const cv::Vec3d P = depth(rng) * (Kinv * cv::Vec3d(a.x, a.y, 1));
        const cv::Vec3d q = K * (R * P + t);
        if (q[2] <= 0) continue;
        const cv::Point2d b(q[0] / q[2], q[1] / q[2]);
        if (b.x < 0 || b.y < 0 || b.x >= size.width || b.y >= size.height) continue;
        pp.cleanA.push_back(a);
        pp.cleanB.push_back(b);
        pp.a.push_back(a + cv::Point2d(g(rng), g(rng)));
        pp.b.push_back(b + cv::Point2d(g(rng), g(rng)));
        pp.isInlier.push_back(1);
    }
    for (int i = 0; i < nOutliers; ++i) {
        pp.a.emplace_back(ux(rng), uy(rng));
        pp.b.emplace_back(ux(rng), uy(rng));
        pp.isInlier.push_back(0);
    }
    return pp;
}

}  // namespace vl::geometry::scenes
