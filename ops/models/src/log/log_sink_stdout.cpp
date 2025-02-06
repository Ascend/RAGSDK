/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "atb_speed/log/log_sink_stdout.h"
#include <iostream>
#include <iomanip>

namespace atb_speed {
LogSinkStdout::LogSinkStdout(LogLevel level) : LogSink(level) {}
const int MICROSECOND = 1000000;
void LogSinkStdout::LogImpl(const LogEntity &logEntity)
{
    std::time_t tmpTime = std::chrono::system_clock::to_time_t(logEntity.time);
    int us =
        std::chrono::duration_cast<std::chrono::microseconds>(logEntity.time.time_since_epoch()).count() % MICROSECOND;
    std::cout << "[" << std::put_time(std::localtime(&tmpTime), "%F %T") << "." << us << "] [" <<
        LogLevelToString(logEntity.level) << "] [" << logEntity.processId << "] [" << logEntity.threadId << "] [" <<
        logEntity.fileName << ":" << logEntity.line << "]" << logEntity.content << std::endl;
}
} // namespace atb