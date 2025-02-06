/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_LOG_LOGCORE_H
#define ATB_SPEED_LOG_LOGCORE_H
#include <memory>
#include <vector>
#include "atb_speed/log/log_entity.h"
#include "atb_speed/log/log_sink.h"
#include "atb/svector.h"

namespace atb_speed {
class LogCore {
public:
	LogCore();
    ~LogCore() = default;
    static LogCore &Instance();
    LogLevel GetLogLevel() const;
    void SetLogLevel(LogLevel level);
    void Log(const LogEntity &logEntity);
    void AddSink(const std::shared_ptr<LogSink> sink);
    const std::vector<std::shared_ptr<LogSink>> &GetAllSinks() const;
    atb::SVector<uint64_t> GetLogLevelCount() const;

private:
	std::vector<std::shared_ptr<LogSink>> sinks_;
    LogLevel level_ = LogLevel::INFO;
    atb::SVector<uint64_t> levelCounts_;
};
} // namespace atb_speed
#endif
