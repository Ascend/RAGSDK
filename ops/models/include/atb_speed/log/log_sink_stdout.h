/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_LOG_LOGSINKSTDOUT_H
#define ATB_SPEED_LOG_LOGSINKSTDOUT_H
#include "atb_speed/log/log_sink.h"

namespace atb_speed {
class LogSinkStdout : public LogSink {
public:
    explicit LogSinkStdout(LogLevel level);
    ~LogSinkStdout() override = default;

private:
    void LogImpl(const LogEntity &logEntity) override;
};
} // namespace atb_speed
#endif