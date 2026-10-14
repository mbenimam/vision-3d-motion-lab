#include "core/scanner/synthetic_page.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "core/common/cv_compat.hpp"
#include "core/common/synthetic.hpp"

namespace vl::scanner {
namespace {

const cv::Scalar kInk(38, 34, 32), kGreyInk(110, 105, 100), kBlue(150, 82, 30), kRed(60, 60, 200);
constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;

// Write `text` in lines no wider than `width`, starting at the baseline y; returns the next baseline.
int paragraph(cv::Mat& page, const std::string& text, int x, int y, int width, double scale, int lineHeight) {
    std::istringstream words(text);
    std::string word, line;
    auto flush = [&] {
        cv::putText(page, line, {x, y}, kFont, scale, kInk, 1, cv::LINE_AA);
        y += lineHeight;
        line.clear();
    };
    while (words >> word) {
        const std::string candidate = line.empty() ? word : line + " " + word;
        int base = 0;
        if (cv::getTextSize(candidate, kFont, scale, 1, &base).width > width && !line.empty()) flush();
        line = line.empty() ? word : line + " " + word;
    }
    if (!line.empty()) flush();
    return y + lineHeight / 2;
}

// Figure 1: a square grid and the same grid seen in perspective, with an arrow between them.
void figure(cv::Mat& page, cv::Rect box) {
    cv::rectangle(page, box, cv::Scalar(236, 238, 240), cv::FILLED);
    cv::rectangle(page, box, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    const cv::Point2f o(float(box.x + 40), float(box.y + 30));
    const float s = 150;
    for (int i = 0; i <= 4; ++i) {
        const float t = s * float(i) / 4;
        cv::line(page, o + cv::Point2f(t, 0), o + cv::Point2f(t, s), kBlue, 2, cv::LINE_AA);
        cv::line(page, o + cv::Point2f(0, t), o + cv::Point2f(s, t), kBlue, 2, cv::LINE_AA);
    }
    // The same grid through a keystone homography: top edge shorter than the bottom one.
    const cv::Point2f q0(float(box.x + box.width - 230), float(box.y + 30));
    auto warp = [&](float u, float v) {
        const float shrink = 0.55f + 0.45f * v;
        return q0 + cv::Point2f(95 + (u - 0.5f) * 190 * shrink, 150 * v * (0.8f + 0.2f * v));
    };
    for (int i = 0; i <= 4; ++i) {
        const float t = float(i) / 4;
        cv::line(page, warp(t, 0), warp(t, 1), kRed, 2, cv::LINE_AA);
        cv::line(page, warp(0, t), warp(1, t), kRed, 2, cv::LINE_AA);
    }
    const int ay = box.y + 30 + 75;
    cv::arrowedLine(page, {box.x + 215, ay}, {box.x + box.width - 255, ay}, kInk, 2, cv::LINE_AA, 0, 0.15);
    cv::putText(page, "H", {box.x + box.width / 2 - 12, ay - 14}, cv::FONT_HERSHEY_DUPLEX, 1.0, kInk, 2, cv::LINE_AA);
}

// A three-column table with ruled lines.
int table(cv::Mat& page, int x, int y, int width) {
    const char* rows[][3] = {{"Points", "Equations", "Unknowns"},
                             {"1 pair", "2", "8 (H up to scale)"},
                             {"4 pairs", "8", "8: exact"},
                             {"n > 4 pairs", "2n", "least squares"}};
    const int rowHeight = 30, col = width / 3;
    cv::line(page, {x, y}, {x + width, y}, kInk, 2, cv::LINE_AA);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 3; ++c)
            cv::putText(page, rows[r][c], {x + 8 + c * col, y + 21 + r * rowHeight},
                        r == 0 ? cv::FONT_HERSHEY_DUPLEX : kFont, 0.58, kInk, 1, cv::LINE_AA);
        if (r == 0) cv::line(page, {x, y + rowHeight}, {x + width, y + rowHeight}, kGreyInk, 1, cv::LINE_AA);
    }
    cv::line(page, {x, y + 4 * rowHeight + 6}, {x + width, y + 4 * rowHeight + 6}, kInk, 2, cv::LINE_AA);
    return y + 4 * rowHeight + 40;
}

// Wood: rings from a sine of the row, bent by smooth noise; 8-bit BGR.
cv::Mat woodDesk(cv::Size size, cv::Scalar tone, std::uint32_t seed) {
    cv::Mat n;
    vl::synth::texture(size, seed).convertTo(n, CV_32F, 1.0 / 255);
    cv::GaussianBlur(n, n, {0, 0}, 14);
    cv::Mat desk(size, CV_8UC3);
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x) {
            const double ring = 0.5 + 0.5 * std::sin(2 * CV_PI * (y / 21.0 + 1.6 * n.at<float>(y, x) + x / 1400.0));
            const double shade = 0.78 + 0.22 * ring + 0.15 * (n.at<float>(y, x) - 0.5);
            desk.at<cv::Vec3b>(y, x) = {cv::saturate_cast<uchar>(tone[0] * shade),
                                        cv::saturate_cast<uchar>(tone[1] * shade),
                                        cv::saturate_cast<uchar>(tone[2] * shade)};
        }
    return desk;
}

// A pale laminate: nearly flat light grey with faint speckle.
cv::Mat paleDesk(cv::Size size, std::uint32_t seed) {
    cv::Mat n;
    vl::synth::texture(size, seed).convertTo(n, CV_32F, 1.0 / 255);
    cv::Mat desk(size, CV_8UC3);
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x) {
            const double v = 0.94 + 0.06 * n.at<float>(y, x);
            desk.at<cv::Vec3b>(y, x) = {cv::saturate_cast<uchar>(206 * v), cv::saturate_cast<uchar>(210 * v),
                                        cv::saturate_cast<uchar>(212 * v)};
        }
    return desk;
}

// A mug seen from above, with its shadow, centred at c.
void mug(cv::Mat& photo, cv::Point c, int r) {
    cv::Mat shade = cv::Mat::zeros(photo.size(), CV_32F);
    cv::circle(shade, c + cv::Point(18, 22), r + 6, cv::Scalar(1), cv::FILLED, cv::LINE_AA);
    cv::GaussianBlur(shade, shade, {0, 0}, 14);
    for (int y = 0; y < photo.rows; ++y)
        for (int x = 0; x < photo.cols; ++x) photo.at<cv::Vec3b>(y, x) *= 1.0 - 0.45 * shade.at<float>(y, x);
    cv::ellipse(photo, c + cv::Point(r - 4, 6), {r / 3, r / 6}, 0, -90, 90, cv::Scalar(225, 228, 232), 14, cv::LINE_AA);
    cv::circle(photo, c, r, cv::Scalar(222, 226, 230), cv::FILLED, cv::LINE_AA);
    cv::circle(photo, c, r - 12, cv::Scalar(30, 48, 70), cv::FILLED, cv::LINE_AA);
    cv::circle(photo, c - cv::Point(r / 4, r / 4), r / 5, cv::Scalar(70, 95, 120), cv::FILLED, cv::LINE_AA);
}

void clutter(cv::Mat& photo) {
    // A pen on the desk, right of the page.
    cv::line(photo, {900, 420}, {870, 900}, cv::Scalar(120, 60, 25), 16, cv::LINE_AA);
    cv::line(photo, {897, 425}, {868, 880}, cv::Scalar(200, 150, 110), 3, cv::LINE_AA);
    cv::line(photo, {870, 900}, {868, 935}, cv::Scalar(180, 180, 180), 10, cv::LINE_AA);
    // A sticky note at the top left, slightly turned.
    const cv::RotatedRect note({{95, 95}, {120, 120}, 12});
    cv::Point2f p[4];
    note.points(p);
    std::vector<cv::Point> poly(p, p + 4);
    cv::fillConvexPoly(photo, poly, cv::Scalar(90, 225, 245), cv::LINE_AA);
    mug(photo, {775, 1150}, 105);  // over the bottom-right corner of the page
}

}  // namespace

cv::Mat renderPage(cv::Size size) {
    cv::Mat page(size, CV_8UC3, cv::Scalar(246, 248, 249));
    const int left = 70, width = size.width - 2 * left;
    cv::putText(page, "Homographies and the DLT", {left, 120}, cv::FONT_HERSHEY_DUPLEX, 1.45, kBlue, 2, cv::LINE_AA);
    cv::putText(page, "3D Vision and Motion  -  lecture notes, part 2", {left, 160},
                cv::FONT_HERSHEY_COMPLEX_SMALL | cv::FONT_ITALIC, 0.95, kGreyInk, 1, cv::LINE_AA);
    cv::line(page, {left, 180}, {left + width, 180}, kGreyInk, 1, cv::LINE_AA);
    int y = paragraph(page,
                      "A plane seen by two cameras, or any scene seen by a camera that only turns about its centre, "
                      "gives two images related by a homography: a 3x3 matrix H, defined up to scale, that sends "
                      "each point x of the first image to the point x' = Hx of the second. It has eight degrees of "
                      "freedom, and each correspondence gives two equations, so four points in general position "
                      "determine it exactly.",
                      left, 225, width, 0.62, 29);
    figure(page, {left, y, width, 215});
    y += 245;
    cv::putText(page, "Figure 1. A square grid and its image under a homography.", {left, y},
                cv::FONT_HERSHEY_COMPLEX_SMALL | cv::FONT_ITALIC, 0.85, kGreyInk, 1, cv::LINE_AA);
    y = paragraph(page,
                  "Writing x' x Hx = 0 turns each pair into two linear equations in the nine entries of H. Stacked, "
                  "they form Ah = 0, and the solution of least algebraic error is the right singular vector of A "
                  "for its smallest singular value. Normalise the points first: centred, at mean distance sqrt(2).",
                  left, y + 50, width, 0.62, 29);
    y = table(page, left, y, width);
    y = paragraph(page,
                  "A document scanner uses the same idea backwards: the four corners of the page and the four "
                  "corners of a rectangle fix H, and every pixel of the flat page is read from the photo at "
                  "H^-1 of its position.",
                  left, y, width, 0.62, 29);
    cv::putText(page, "Exercises", {left, y + 10}, cv::FONT_HERSHEY_DUPLEX, 0.8, kBlue, 1, cv::LINE_AA);
    const char* items[] = {"1. Why does a homography have 8 degrees of freedom and not 9?",
                           "2. Which configurations of 4 points do not determine H?",
                           "3. Show that H maps lines to lines and keeps cross-ratios."};
    for (int i = 0; i < 3; ++i)
        cv::putText(page, items[i], {left + 20, y + 50 + 30 * i}, kFont, 0.62, kInk, 1, cv::LINE_AA);
    cv::putText(page, "- 2 -", {size.width / 2 - 22, size.height - 50}, kFont, 0.6, kGreyInk, 1, cv::LINE_AA);
    return page;
}

DocumentPhoto documentPhoto(DeskScene scene, std::uint32_t seed) {
    const cv::Size photoSize(960, 1280);
    DocumentPhoto d;
    d.page = renderPage();
    const double W = d.page.cols, H = d.page.rows;
    switch (scene) {
        case DeskScene::Easy:
            d.corners = {{{170, 190}, {800, 168}, {832, 1088}, {142, 1116}}};
            break;
        case DeskScene::StrongPerspective:
            d.corners = {{{262, 250}, {712, 236}, {908, 1156}, {66, 1140}}};
            break;
        case DeskScene::Shadowed:
            d.corners = {{{150, 170}, {792, 205}, {818, 1112}, {118, 1082}}};
            break;
        case DeskScene::LowContrast:
            d.corners = {{{182, 212}, {782, 190}, {812, 1080}, {160, 1102}}};
            break;
        case DeskScene::Cluttered:
            d.corners = {{{226, 176}, {836, 262}, {734, 1146}, {108, 1046}}};
            break;
    }
    // The homography taking the page's outer pixel edges onto the chosen corners.
    const cv::Point2f src[4] = {
        {-0.5f, -0.5f}, {float(W) - 0.5f, -0.5f}, {float(W) - 0.5f, float(H) - 0.5f}, {-0.5f, float(H) - 0.5f}};
    cv::Point2f dst[4];
    for (int i = 0; i < 4; ++i) dst[i] = cv::Point2f(d.corners[size_t(i)]);
    const cv::Mat Hgt = cv::getPerspectiveTransform(src, dst);

    cv::Mat desk =
        scene == DeskScene::LowContrast
            ? paleDesk(photoSize, seed)
            : woodDesk(photoSize, scene == DeskScene::Shadowed ? cv::Scalar(45, 70, 100) : cv::Scalar(60, 105, 150),
                       seed);
    cv::Mat warped, mask;
    cv::warpPerspective(d.page, warped, Hgt, photoSize, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::warpPerspective(cv::Mat(d.page.size(), CV_32F, cv::Scalar(1)), mask, Hgt, photoSize, cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT);
    // The page's own soft shadow on the desk, then the page on top (the mask is anti-aliased).
    cv::Mat drop;
    cv::warpAffine(mask, drop, cv::Matx23d(1, 0, 6, 0, 1, 10), photoSize);
    cv::GaussianBlur(drop, drop, {0, 0}, 9);
    cv::Mat photo(photoSize, CV_32FC3);
    for (int y = 0; y < photoSize.height; ++y)
        for (int x = 0; x < photoSize.width; ++x) {
            const float m = mask.at<float>(y, x);
            const cv::Vec3f desk3 = cv::Vec3f(desk.at<cv::Vec3b>(y, x)) * (1.f - 0.35f * drop.at<float>(y, x));
            photo.at<cv::Vec3f>(y, x) = desk3 * (1.f - m) + cv::Vec3f(warped.at<cv::Vec3b>(y, x)) * m;
        }
    // Light: a lamp at the top left (falloff), warm; a soft hand-shaped shadow for the shadowed scene.
    cv::Mat shadow = cv::Mat::zeros(photoSize, CV_32F);
    if (scene == DeskScene::Shadowed) {
        cv::ellipse(shadow, {700, 330}, {330, 180}, 35, 0, 360, cv::Scalar(1), cv::FILLED);
        cv::ellipse(shadow, {960, 120}, {200, 120}, 35, 0, 360, cv::Scalar(1), cv::FILLED);
        cv::GaussianBlur(shadow, shadow, {0, 0}, 45);
    }
    const double falloff = scene == DeskScene::Shadowed ? 0.45 : scene == DeskScene::LowContrast ? 0.12 : 0.25;
    for (int y = 0; y < photoSize.height; ++y)
        for (int x = 0; x < photoSize.width; ++x) {
            const double r = std::hypot(x + 150.0, y + 150.0) / std::hypot(1110.0, 1430.0);
            const double light = (1.04 - falloff * r * r) * (1 - 0.55 * shadow.at<float>(y, x));
            const cv::Vec3f warm(0.93f, 0.98f, 1.0f);
            cv::Vec3f& v = photo.at<cv::Vec3f>(y, x);
            for (int c = 0; c < 3; ++c) v[c] = float(v[c] * light * warm[c]);
        }
    photo.convertTo(d.photo, CV_8UC3);
    if (scene == DeskScene::Cluttered) clutter(d.photo);
    cv::GaussianBlur(d.photo, d.photo, {0, 0}, 0.9);  // the lens is not perfect
    cv::RNG rng(seed);
    vl::synth::addNoise(d.photo, scene == DeskScene::LowContrast ? 3.0 : 2.5, rng);
    return d;
}

}  // namespace vl::scanner
