/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_PLUGIN_ACLNN_NN_TENSOR_H
#define ATB_SPEED_PLUGIN_ACLNN_NN_TENSOR_H
#include <string>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <atb/atb_infer.h>

namespace atb_speed {
namespace common {

using UpdateDataPtrFunc = std::function<aclnnStatus(aclOpExecutor *, const size_t, aclTensor *, void *)>;

class AclNNTensor {
public:
    enum ParallelType : int {
        NOT_IN_TENSORLIST = -1,
    };
    atb::Tensor atbTensor;
    atb::SVector<int64_t> strides;
    aclTensor *tensor = nullptr;
    int tensorListidx = NOT_IN_TENSORLIST; // tensorlist在cache和executor中的index，需要填充nullptr保持两边inedx保持一致
    int tensorIdx = -1;  // aclTensor在aclExecutor中的index
    bool needUpdateTensorDataPtr = false;
};

} // namespace common
} // namespace atb_speed
#endif