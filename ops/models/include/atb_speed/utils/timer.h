/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_UTILS_TIMER_H
#define ATB_SPEED_UTILS_TIMER_H
#include <cstdint>

namespace atb_speed {
class Timer {
public:
    Timer();
    ~Timer();
    uint64_t ElapsedMicroSecond();
    void Reset();

private:
    uint64_t GetCurrentTimepoint() const;

private:
    uint64_t startTimepoint_ = 0;
};
} // namespace atb_speed
#endif