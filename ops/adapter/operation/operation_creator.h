/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#ifndef ATB_SPEED_OPERATION_CREATOR_H
#define ATB_SPEED_OPERATION_CREATOR_H
#include <string>
#include "atb/infer_op_params.h"
#include "atb/operation.h"

namespace atb_speed {
    atb::Operation *CreateOperation(const std::string &opName, const std::string &param);
} /* namespace atb */

#endif