/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_HOSTTENSOR_BINDER_H
#define ATB_SPEED_HOSTTENSOR_BINDER_H
#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>

namespace atb_speed {
class HostTensorBinder {
public:
    HostTensorBinder() = default;
    virtual ~HostTensorBinder() = default;
    virtual void ParseParam(const nlohmann::json &paramJson) = 0;
    virtual void BindTensor(atb::VariantPack &variantPack) = 0;
};
} // namespace atb_speed
#endif