/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_UTILS_WORKSPACE_H
#define ATB_SPEED_UTILS_WORKSPACE_H
#include <cstdint>
#include <memory>
#include <vector>
#include "buffer_base.h"

namespace atb_speed {

class Workspace {
public:
    Workspace();
    ~Workspace();
    void *GetWorkspaceBuffer(uint64_t bufferSize);

private:
    uint64_t GetWorkspaceBufferRing() const;
    uint64_t GetWorkspaceBufferSize() const;

private:
    std::vector<std::unique_ptr<BufferBase>> workspaceBuffers_;
    size_t workspaceBufferOffset_ = 0;
};
} // namespace atb_speed
#endif