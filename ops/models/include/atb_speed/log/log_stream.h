/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_LOG_LOGSTREAM_H
#define ATB_SPEED_LOG_LOGSTREAM_H
#include <sstream>
#include <vector>
#include <iostream>
#include "atb_speed/log/log_entity.h"

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    for (auto& el : vec) {
        os << el << ',';
    }
    return os;
}

namespace atb_speed {
class LogStream {
public:
    LogStream(const char *filePath, int line, const char *funcName, LogLevel level);
    ~LogStream();
    friend std::ostream& operator<<(std::ostream& os, const LogStream& obj);
    template <typename T> LogStream &operator << (const T &value)
    {
        stream_ << value;
        return *this;
    }
    void Format(const char *format, ...);

private:
    LogEntity logEntity_;
    std::stringstream stream_;
    bool useStream_ = true;
};
} // namespace atb_speed
#endif
