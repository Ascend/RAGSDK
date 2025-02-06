/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_GLOBAL_CACHE_H
#define ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_GLOBAL_CACHE_H

#include <vector>
#include <string>
#include <map>
#include <atb/atb_infer.h>
#include "acl_nn_operation_cache.h"
#include "acl_nn_operation.h"

namespace atb_speed {
namespace common {

const uint16_t DEFAULT_ACLNN_GLOBAL_CACHE_SIZE = 16;
constexpr int32_t DECIMAL = 10;

class AclNNGlobalCache {
public:
    explicit AclNNGlobalCache();
    std::shared_ptr<AclNNOpCache> GetGlobalCache(std::string opName, atb::VariantPack variantPack);
    atb::Status UpdateGlobalCache(std::string opName, std::shared_ptr<AclNNOpCache> cache);
    std::string PrintGlobalCache();

private:
    int nextUpdateIndex_ = 0;
    uint16_t globalCacheCountMax_ = 16;
    std::map<std::string, std::vector<std::shared_ptr<AclNNOpCache>>> aclnnGlobalCache_;
};

} // namespace common
} // namespace atb_speed
#endif