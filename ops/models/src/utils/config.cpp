/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "atb_speed/utils/config.h"
#include <string>
#include <unistd.h>
#include <iostream>
#include <thread>
#include <atb_speed/utils/match.h>
#include <atb_speed/utils/str_split.h>
#include "atb_speed/log.h"

namespace atb_speed {
Config::Config()
{
    InitSaveTensor();
    isSaveTensor_ = IsEnable();
    isConvertNCHWToND_ = IsEnable();
    isTorchTensorFormatCast_ = IsEnable();
    isUseTilingCopyStream_ = IsEnable();
    isLayerInternalTensorReuse_ = IsEnable();
    ATB_LOG(FATAL) << "Config:\nIsSaveTensor:" << isSaveTensor_ << " \nIsConvertNCHWToND:" << isConvertNCHWToND_
                   << "\nIsTorchTensorFormatCast:" << isTorchTensorFormatCast_
                   << "\nIsLayerInternalTensorReuse:" << isLayerInternalTensorReuse_;
}

Config::~Config() {}

std::string Config::GetSaveTensorDir()
{
    std::ostringstream pid;
    pid << getpid();
   
    return "tensors/thread_" + pid.str();
}

bool Config::IsEnable(bool enable)
{
    return enable;
}

bool Config::IsSaveTensor() const { return isSaveTensor_; }

void Config::DisableSaveTensor() { isSaveTensor_ = false; }

uint64_t Config::GetSaveTensorMaxNum() const { return saveTensorMaxNum_; }

bool Config::IsTorchTensorFormatCast() const { return isTorchTensorFormatCast_; };

bool Config::IsConvertNCHWToND() const { return isConvertNCHWToND_; }

bool Config::IsUseTilingCopyStream() const { return isUseTilingCopyStream_; }

bool Config::IsSaveTensorForRunner(const std::string &runnerName) const
{
    if (saveTensorRunnerNameSet_.empty()) {
        return true;
    }

    for (auto &name : saveTensorRunnerNameSet_) {
        if (atb_speed::StartsWith(runnerName, name)) {
            return true;
        }
    }
    return false;
}

void Config::InitSaveTensor() const
{
    return;
}

bool Config::IsLayerInternalTensorReuse() const
{
    return isLayerInternalTensorReuse_;
}
} // namespace atb_speed