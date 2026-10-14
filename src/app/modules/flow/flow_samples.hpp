#pragma once
// Inputs shared by the optical-flow topics: two synthetic pairs with exact flow (always there), then
// the Middlebury "other" sequences that come with ground truth, when scripts/fetch_data.sh has run.
//
// Inputs::images = {frame 0, frame 1}; Inputs::truth = {flow CV_32FC2, frame 0 (x, y) -> frame 1
// (x + u, y + v); |u| > 1e9 marks unknown pixels, as in .flo files}.

#include <vector>

#include "app/module.hpp"

namespace vl::app::flow_samples {

std::vector<Sample> all();

}  // namespace vl::app::flow_samples
