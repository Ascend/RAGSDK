/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "workspace.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/config.h"
#include "buffer_device.h"

namespace atb_speed {

Workspace::Workspace()
{
    uint64_t bufferRing = GetWorkspaceBufferRing();
    uint64_t bufferSize = GetWorkspaceBufferSize();
    ATB_LOG(FATAL) << "Workspace workspace bufferRing:" << bufferRing << ", bufferSize:" << bufferSize;
    workspaceBuffers_.resize(bufferRing);
    for (size_t i = 0; i < bufferRing; ++i) {
        workspaceBuffers_.at(i).reset(new BufferDevice(bufferSize));
    }
}

Workspace::~Workspace() {}

void *Workspace::GetWorkspaceBuffer(uint64_t bufferSize)
{
    if (workspaceBufferOffset_ == workspaceBuffers_.size()) {
        workspaceBufferOffset_ = 0;
    }
    return workspaceBuffers_.at(workspaceBufferOffset_++)->GetBuffer(bufferSize);
}

uint64_t Workspace::GetWorkspaceBufferRing() const
{
    return 1;
}

uint64_t Workspace::GetWorkspaceBufferSize() const
{
    return 0;
}

} // namespace atb_speed