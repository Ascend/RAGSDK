/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_UTILS_STRINGS_STRSPLIT_H
#define ATB_SPEED_UTILS_STRINGS_STRSPLIT_H
#include <string>
#include <vector>

namespace atb_speed {
void StrSplit(const std::string &text, const char delimiter, std::vector<std::string> &result);
std::string GetFuncNameAndNameSpace(const std::string &inputStr);
} // namespace atb_speed
#endif