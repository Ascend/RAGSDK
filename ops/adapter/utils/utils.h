/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#ifndef EXAMPLE_UTIL_H
#define EXAMPLE_UTIL_H
#include <vector>
#include <string>
#include <atb/types.h>
#include <torch/torch.h>
#include "atb/operation.h"

namespace atb_speed {
class Utils {
public:
    static void *GetCurrentStream();
    static int64_t GetTensorNpuFormat(const at::Tensor &tensor);
    static at::Tensor NpuFormatCast(const at::Tensor &tensor);
    static void BuildVariantPack(const std::vector<torch::Tensor> &inTensors,
                                 const std::vector<torch::Tensor> &outTensors, atb::VariantPack &variantPack);
    static atb::Tensor AtTensor2Tensor(const at::Tensor &atTensor);
    static at::Tensor CreateAtTensorFromTensorDesc(const atb::TensorDesc &tensorDesc);
    static void ContiguousAtTensor(std::vector<torch::Tensor> &atTensors);
    static void ContiguousAtTensor(torch::Tensor &atTensor);
};
} // namespace atb_speed

#endif