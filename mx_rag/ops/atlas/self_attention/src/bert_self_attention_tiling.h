/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#ifndef BERT_SELF_ATTENTION_H
#define BERT_SELF_ATTENTION_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"


struct BertSelfAttentionTilingData {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t headNum;
    uint32_t headDim;
    uint32_t resverSize; // 32k byte
    uint32_t seqProcSlice;
    float rsqrtHeadDim;
    uint64_t l1BufferSize;
    SoftMaxTiling softMaxTilingData;
};

int GenerateTiling(const char *socVersion, BertSelfAttentionTilingData &tiling);

#endif