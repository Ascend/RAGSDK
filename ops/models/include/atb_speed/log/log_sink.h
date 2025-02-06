/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_LOG_LOGSINK_H
#define ATB_SPEED_LOG_LOGSINK_H
#include "atb_speed/log/log_entity.h"

namespace atb_speed {
class LogSink {
public:
    explicit LogSink(LogLevel level = LogLevel::INFO);
    virtual ~LogSink() = default;
    void Log(const LogEntity &logEntity);

private:
    virtual void LogImpl(const LogEntity &logEntity) = 0;
    LogLevel level_;
};
} // namespace atb_speed
#endif