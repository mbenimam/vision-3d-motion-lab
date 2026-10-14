#pragma once
// OpenCV 4 and 5 put the same functions in differently named modules. Include this instead of the
// module headers when you call OpenCV's own methods: the baselines our implementations are compared to.

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#if __has_include(<opencv2/features.hpp>)
#include <opencv2/features.hpp>  // OpenCV 5: cv::SIFT, cv::BFMatcher
#else
#include <opencv2/features2d.hpp>  // OpenCV 4
#endif

#if __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>  // OpenCV 5: findHomography, findFundamentalMat
#else
#include <opencv2/calib3d.hpp>  // OpenCV 4
#endif

#if __has_include(<opencv2/stereo.hpp>)
#include <opencv2/stereo.hpp>  // OpenCV 5: StereoBM, StereoSGBM (OpenCV 4 has them in calib3d, included above)
#endif

#include <opencv2/video.hpp>  // OpenCV baselines: optical flow, background subtraction, meanShift
