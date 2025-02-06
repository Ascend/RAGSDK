/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "atb_speed/log/log_entity.h"
#include <vector>
#include <string>

namespace atb_speed {
std::string LogLevelToString(LogLevel level)
{
	static std::vector<std::string> levelStrs = { "trace", "debug", "info", "warn", "error", "fatal" };
    size_t levelInt = static_cast<size_t>(level);
    return levelInt < levelStrs.size() ? levelStrs[levelInt] : "unknown";
}
} // namespace atb