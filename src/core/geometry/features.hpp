#pragma once
// Putative correspondences between two images: SIFT keypoints and descriptors (OpenCV), matched
// by our own brute-force nearest-neighbour search with Lowe's ratio test and a mutual check.
//
// Ratio test: a match is kept when its nearest descriptor is clearly closer than the second nearest,
// d1 / d2 < ratio. Repetitive texture gives two similar candidates and fails the test.
// Mutual check: the nearest neighbour of b must be a again (cross-consistency).
//
// Exhaustive search: all squared distances, by blocks of rows of a (cv::batchDistance as the vectorised
// primitive), each block scanned once for the two nearest of every row and the nearest of every
// column. O(n1 · n2 · 128) flops, O(block · n2) memory, blocks in parallel.

#include <cstdint>
#include <deque>
#include <mutex>
#include <stop_token>
#include <vector>

#include <opencv2/core.hpp>

namespace vl::geometry {

struct Features {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;  // CV_32F, one 128-d row per keypoint
};

// SIFT on the grey image, keeping the maxFeatures strongest responses.
Features detectSift(const cv::Mat& image, int maxFeatures = 3000);

struct Match {
    int a = -1, b = -1;   // indices in the first / second feature set
    float distance = 0;   // descriptor distance to the nearest neighbour
    float ratio = 0;      // nearest / second nearest
};

struct MatchParams {
    float ratio = 0.75f;  // Lowe's threshold; 1 disables the test
    bool mutual = true;
};

std::vector<Match> matchDescriptors(const cv::Mat& da, const cv::Mat& db, const MatchParams& p,
                                    std::stop_token stop = {});

// Remembers the features of the last few images (keyed by a hash of their pixels), so moving a
// slider that does not change the images skips detection. Thread-safe.
class FeatureCache {
public:
    Features get(const cv::Mat& image, int maxFeatures);

private:
    struct Entry {
        std::uint64_t key;
        Features f;
    };
    std::mutex mutex_;
    std::deque<Entry> entries_;
};

}  // namespace vl::geometry
