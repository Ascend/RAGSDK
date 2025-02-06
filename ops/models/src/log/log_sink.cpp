/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "atb_speed/log/log_sink.h"

namespace atb_speed {
LogSink::LogSink(LogLevel level) : level_(level) {}

void LogSink::Log(const LogEntity &logEntity)
{
    if (logEntity.level >= level_) {
        LogImpl(logEntity);
    }
}
} // namespace atb