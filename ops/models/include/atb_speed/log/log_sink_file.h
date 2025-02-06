/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_LOG_SINKFILE_H
#define ATB_SPEED_LOG_SINKFILE_H

#include <fstream>
#include "atb_speed/log/log_sink.h"

namespace atb_speed {
class LogSinkFile : public LogSink {
public:
	explicit LogSinkFile(LogLevel level);
    ~LogSinkFile() override;

private:
	void LogImpl(const LogEntity &logEntity) override;

private:
	std::ofstream fileHandle_;
    int32_t fileCount_ = 0;
    bool isFlush_ = false;
    std::string curTime_;
    std::string fileDir_ = "atb_temp/log/";
};
} // namespace atb_speed
#endif
