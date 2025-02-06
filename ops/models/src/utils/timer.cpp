/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "atb_speed/utils/timer.h"
#include <sys/time.h>

namespace atb_speed {
const uint64_t MICRSECOND_PER_SECOND = 1000000;

Timer::Timer() { startTimepoint_ = GetCurrentTimepoint(); }

Timer::~Timer() {}

uint64_t Timer::ElapsedMicroSecond()
{
    uint64_t now = GetCurrentTimepoint();
    uint64_t use = now - startTimepoint_;
    startTimepoint_ = now;
    return use;
}

void Timer::Reset() { startTimepoint_ = GetCurrentTimepoint(); }

uint64_t Timer::GetCurrentTimepoint() const
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    uint64_t ret =
        static_cast<uint64_t>(tv.tv_sec * MICRSECOND_PER_SECOND + tv.tv_usec);
    return ret;
}
} // namespace atb_speed