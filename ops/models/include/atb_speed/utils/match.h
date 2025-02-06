/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_UTILS_STRINGS_MATCH_H
#define ATB_SPEED_UTILS_STRINGS_MATCH_H
#include <string>

namespace atb_speed {
bool StartsWith(const std::string &text, const std::string &prefix);
bool EndsWith(const std::string &text, const std::string &suffix);
} // namespace atb_speed
#endif