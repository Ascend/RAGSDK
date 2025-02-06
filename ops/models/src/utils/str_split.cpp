/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "atb_speed/utils/str_split.h"
#include <sstream>

namespace atb_speed {
constexpr int OPGRAPH_NAME_MAX_LENG = 128;

void StrSplit(const std::string &text, const char delimiter, std::vector<std::string> &result)
{
    std::istringstream iss(text);
    std::string subStr;
    while (getline(iss, subStr, delimiter)) {
        result.push_back(subStr);
    }
}

std::string GetFuncNameAndNameSpace(const std::string &inputStr)
{
    int spaceInd = 0;
    int leftBracketInd = 0;
    std::string extractStr;
    int inputStrLen = static_cast<int>(inputStr.size());
    for (int i = 0; i < inputStrLen; i++) {
        if (inputStr.at(i) == ' ') {
            spaceInd = i;
        } else if (inputStr.at(i) == '(') {
            leftBracketInd = i;
            break;
        }
    }
    if (spaceInd >= 0 && (leftBracketInd - spaceInd) > 0) {
        int len;
        if (leftBracketInd - (spaceInd + 1) > OPGRAPH_NAME_MAX_LENG) {
            len = OPGRAPH_NAME_MAX_LENG;
        } else {
            len = leftBracketInd - (spaceInd + 1);
        }
        extractStr = inputStr.substr(spaceInd + 1, len);
    } else {
        extractStr = inputStr;
    }

    for (char &i : extractStr) {
        if (!isalnum(i) && i != '_') {
            i = '_';
        }
    }
    return extractStr;
}

} // namespace atb_speed