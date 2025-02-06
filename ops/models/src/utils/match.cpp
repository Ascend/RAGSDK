/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "atb_speed/utils/match.h"
#include <cstring>

namespace atb_speed {
bool StartsWith(const std::string &text, const std::string &prefix)
{
    return prefix.empty() || (text.size() >= prefix.size() && memcmp(text.data(), prefix.data(), prefix.size()) == 0);
}

bool EndsWith(const std::string &text, const std::string &suffix)
{
    return suffix.empty() || (text.size() >= suffix.size() &&
                              memcmp(text.data() + (text.size() - suffix.size()), suffix.data(), suffix.size()) == 0);
}
} // namespace atb_speed
