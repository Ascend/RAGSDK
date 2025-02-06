/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#ifndef ATB_SPEED_LAYERS_FEED_FORWARD_H
#define ATB_SPEED_LAYERS_FEED_FORWARD_H

#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace bge_large {
struct FeedForwardParam {
    void *hcclComm = nullptr;
    atb::infer::ActivationType activationType;
    int64_t geluApproximate = -1;
    bool transposeB = true;
    bool isBias = true;
    std::string backend = "hccl";
    bool isBF16 = false;
};

atb::Status FeedForwardLayer(const FeedForwardParam &param, atb::Operation **operation);
} // namespace bge_large
} // namespace atb_speed
#endif
