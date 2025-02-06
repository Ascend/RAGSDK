/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#ifndef ATB_SPEED_PLUGIN_UTILS_H
#define ATB_SPEED_PLUGIN_UTILS_H
#include "acl_nn_operation.h"

namespace atb_speed {
namespace common {

const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
const int DIM3 = 3;
const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;
const int NUM4 = 4;
const int NUM5 = 5;

atb::SVector<int64_t> GetCopyTensorStride(atb::Dims &tensorDims);

atb::SVector<int64_t> GetTransposeTensorStride(atb::Dims &tensorDims);

bool Is910B();

atb::Tensor SqueezeBatchSeq(atb::Tensor atbTensor);

bool isVariankPackEqual(const AclNNVariantPack &aclnnVariantPack, const atb::VariantPack &atbVariantPack);

} // namespace common
} // namespace atb_speed
#endif