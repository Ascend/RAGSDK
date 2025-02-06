/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_UTILS_STATISTIC_H
#define ATB_SPEED_UTILS_STATISTIC_H
#include <string>

namespace atb_speed {
struct Statistic {
    uint64_t totalTime = 0;
    uint64_t createTensorTime = 0;
    uint64_t planSetupTime = 0;
    uint64_t planAsyncTime = 0;
    uint64_t planExecuteTime = 0;
    uint64_t streamSyncTime = 0;
    uint64_t tillingCopyTime = 0;
    uint64_t getBestKernelTime = 0;
    uint64_t kernelExecuteTime = 0;
    uint64_t kernelCacheHitCount = 0;
    uint64_t kernelCacheMissCount = 0;
    uint64_t mallocTorchTensorSize = 0;

    std::string ToString() const;
    void Reset();
};

Statistic &GetStatistic();
} // namespace atb_speed
#endif