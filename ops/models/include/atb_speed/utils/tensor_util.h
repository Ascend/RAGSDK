/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_UTILS_TENSOR_UTIL_H
#define ATB_SPEED_UTILS_TENSOR_UTIL_H
#include <string>
#include <atb/types.h>

namespace atb_speed {
class TensorUtil {
public:
    static std::string TensorToString(const atb::Tensor &tensor);
    static std::string TensorDescToString(const atb::TensorDesc &tensorDesc);
    static uint64_t GetTensorNumel(const atb::Tensor &tensor);
    static uint64_t GetTensorNumel(const atb::TensorDesc &tensorDesc);
    static bool TensorDescEqual(const atb::TensorDesc &tensorDescA, const atb::TensorDesc &tensorDescB);
};
} // namespace atb_speed
#endif