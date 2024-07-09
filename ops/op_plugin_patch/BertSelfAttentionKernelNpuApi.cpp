/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include <iostream>
#include <vector>
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {
constexpr size_t BASESEQLENGTH = 16;
constexpr size_t ATTENTIONMASKCATAIX = 3;

using npu_preparation = at_npu::native::OpPreparation;
using namespace at_npu::native;

enum class DropOutStatus {
    DROPOUT_NORMAL = 0,
    DROPOUT_NONE,
    DROPOUT_ALL
};

struct BertSelfAttentionParams {
    at::Tensor input;
    at::Tensor input_padded;
    at::Tensor queryW;
    at::Tensor queryBias;
    at::Tensor keyW;
    at::Tensor keyBias;
    at::Tensor valueW;
    at::Tensor valueBias;
    at::Tensor attentionMask;
    at::Tensor attentionMask_padded;
    at::Tensor headMask;
    int64_t numAttentionHeads;
    int64_t attentionHeadSize;
    uint64_t seqLenPadded;
    double dropOutKeepProb;
    at::Tensor dropOutMask;
    at::IntArrayRef inputSizes;
};

DropOutStatus GetDropoutStatus(double &keepProb, bool train)
{
    if (!train) {
        keepProb = 1.0;
    }

    if (keepProb == 0) {
        return DropOutStatus::DROPOUT_ALL;
    }

    if (keepProb == 1) {
        return DropOutStatus::DROPOUT_NONE;
    }

    return DropOutStatus::DROPOUT_NORMAL;
}

void padding_input_and_attentionmask(uint64_t &seqLenPadded, at::Tensor &input_padded, at::Tensor &attentionMask_padded,
                                     const at::Tensor &input, const at::Tensor &attentionMask)
{
    auto input_sizes = input.sizes();
    if (input_sizes[1] % BASESEQLENGTH != 0) {
        int paddingSize = BASESEQLENGTH - input_sizes[1] % BASESEQLENGTH;
        seqLenPadded += paddingSize;
        auto padding_tensor = at::full({ input.size(0), paddingSize, input.size(2) }, 0, input.options());
        input_padded = at::cat({ input, padding_tensor }, 1);
        padding_tensor = at::full({ attentionMask.size(0), attentionMask.size(1), attentionMask.size(2), paddingSize },
                                  std::numeric_limits<float>::lowest(), attentionMask.options());
        attentionMask_padded = at::cat({ attentionMask, padding_tensor }, ATTENTIONMASKCATAIX);
    }
}

int check_tensor_shape(std::vector<at::Tensor> tensors)
{
    std::vector<int> expected_dims = { 3, 2, 1, 2, 1, 2, 1, 4, 4 };

    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].dim() != expected_dims[i]) {
            std::cout << "Tensor at index " + std::to_string(i) +
                             " does not have the expected number of dimensions: expected " +
                             std::to_string(expected_dims[i]) + ", got " + std::to_string(tensors[i].dim())
                      << std::endl;
            return -1;
        }
    }
    return 0;
}

at::Tensor execute_bert_self_attention(const BertSelfAttentionParams &params)
{
    at::Tensor result;

    if (params.inputSizes[1] % BASESEQLENGTH == 0) {
        result = npu_preparation::apply_tensor_without_format(params.input);
        EXEC_NPU_CMD(aclnnBertSelfAttention, params.input, params.queryW, params.queryBias, params.keyW, params.keyBias,
                     params.valueW, params.valueBias, params.attentionMask, params.dropOutMask, params.headMask,
                     params.numAttentionHeads, params.attentionHeadSize, params.dropOutKeepProb, result);
    } else {
        std::vector<int64_t> output_sizes = { params.inputSizes[0], params.seqLenPadded, params.inputSizes[2] };
        c10::IntArrayRef output_size_ref(output_sizes);
        at::Tensor result_padded = npu_preparation::apply_tensor_without_format(output_size_ref,
                                                                                params.input.options());
        EXEC_NPU_CMD(aclnnBertSelfAttention, params.input_padded, params.queryW, params.queryBias, params.keyW,
                     params.keyBias, params.valueW, params.valueBias, params.attentionMask_padded, params.dropOutMask,
                     params.headMask, params.numAttentionHeads, params.attentionHeadSize, params.dropOutKeepProb,
                     result_padded);
        result = result_padded.narrow(1, 0, params.inputSizes[1]);
    }

    return result;
}

at::Tensor npu_bert_self_attention_custom(const at::Tensor &input, const at::Tensor &queryW,
                                          const at::Tensor &queryBias, const at::Tensor &keyW,
                                          const at::Tensor &keyBias, const at::Tensor &valueW,
                                          const at::Tensor &valueBias, const at::Tensor &attentionMask,
                                          const at::Tensor &headMask, int64_t numAttentionHeads,
                                          int64_t attentionHeadSize, double dropOutKeepProb, bool train)
{
    std::vector<at::Tensor> tensors = { input,  queryW,    queryBias,     keyW,    keyBias,
                                        valueW, valueBias, attentionMask, headMask };
    if (check_tensor_shape(tensors) == -1) {
        return at::Tensor();
    }
    double keepProb = 1.0 - dropOutKeepProb;
    auto input_sizes = input.sizes();
    uint64_t seqLenPadded = input_sizes[1];

    // padding input and attentionmask
    at::Tensor input_padded;
    at::Tensor attentionMask_padded;
    padding_input_and_attentionmask(seqLenPadded, input_padded, attentionMask_padded, input, attentionMask);

    int64_t maskLen = (input_sizes[0] * numAttentionHeads * seqLenPadded * seqLenPadded + 128 - 1) / 128 * 128 / 8;

    // gen_drop_out_mask
    at::Tensor dropOutMask;
    DropOutStatus droupOutStatus = GetDropoutStatus(keepProb, train);
    if (droupOutStatus == DropOutStatus::DROPOUT_ALL) {
        dropOutMask = at::zeros(at::IntArrayRef{maskLen}, input.options().dtype(at::kByte));
    } else if (droupOutStatus == DropOutStatus::DROPOUT_NONE) {
        dropOutKeepProb = 0;
        dropOutMask = at::ones(at::IntArrayRef{maskLen}, input.options().dtype(at::kByte));
    } else {
        auto original_stream = c10_npu::getCurrentNPUStream();
        c10_npu::SecondaryStreamGuard guard(c10_npu::getCurrentSecondaryStream());
        dropOutMask = at_npu::native::OpPreparation::apply_tensor_without_format({maskLen},
                                                                                 input.options().dtype(at::kByte));
        std::vector<int64_t> shape = { input_sizes[0], numAttentionHeads, seqLenPadded, seqLenPadded };
        at::IntArrayRef shapeArray(shape);

        const auto gen = at_npu::detail::getDefaultNPUGenerator();
        auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(10);

        const uint64_t seed = pair.first;
        const uint64_t offset = pair.second;
        EXEC_NPU_CMD(aclnnDropoutGenMask, shapeArray, dropOutKeepProb, seed, offset, dropOutMask);
        c10_npu::NPUCachingAllocator::recordStream(dropOutMask.storage().data_ptr(), original_stream);
    }

    BertSelfAttentionParams params = { input, input_padded, queryW, queryBias, keyW, keyBias, valueW,
                                       valueBias, attentionMask, attentionMask_padded, headMask,
                                       numAttentionHeads, attentionHeadSize, seqLenPadded,
                                       dropOutKeepProb, dropOutMask, input_sizes };

    return execute_bert_self_attention(params);
}
}  // namespace op_api
