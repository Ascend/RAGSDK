/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#ifndef ATB_SPEED_CONTEXT_BUFFER_BASE_H
#define ATB_SPEED_CONTEXT_BUFFER_BASE_H
#include <cstdint>

namespace atb_speed {
class BufferBase {
public:
    BufferBase();
    virtual ~BufferBase();
    virtual void *GetBuffer(uint64_t bufferSize) = 0;
};
} // namespace atb_speed
#endif