/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#ifndef ATB_SPEED_CONTEXT_BUFFER_DEVICE_H
#define ATB_SPEED_CONTEXT_BUFFER_DEVICE_H
#include <torch/torch.h>
#include "buffer_base.h"

namespace atb_speed {
class BufferDevice : public BufferBase {
public:
    explicit BufferDevice(uint64_t bufferSize);
    ~BufferDevice() override;
    void *GetBuffer(uint64_t bufferSize) override;
private:
    torch::Tensor CreateAtTensor(const uint64_t bufferSize) const;

private:
    void *buffer_ = nullptr;
    uint64_t bufferSize_ = 0;
    torch::Tensor atTensor_;
};
} // namespace atb_speed
#endif