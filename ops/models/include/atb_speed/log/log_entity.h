/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_LOG_LOGENTITY_H
#define ATB_SPEED_LOG_LOGENTITY_H
#include <chrono>
#include <string>

namespace atb_speed {
enum class LogLevel {
    TRACE = 0,
	DEBUG,
	INFO,
	WARN,
	ERROR,
	FATAL
};

std::string LogLevelToString(LogLevel level);

struct LogEntity {
    std::chrono::system_clock::time_point time;
    size_t processId = 0;
    size_t threadId = 0;
    LogLevel level = LogLevel::TRACE;
    const char *fileName = nullptr;
    int line = 0;
    const char *funcName = nullptr;
    std::string content;
};
} // namespace atb_speed
#endif