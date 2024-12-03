/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <cmath>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "./bert_self_attention_tiling.h"

using namespace matmul_tiling;
using namespace std;

constexpr int64_t RESVER_SIZE = 175 * 1024;
constexpr uint32_t MAX_SEQ_LEN = 1024;
constexpr uint32_t MAX_SEQ_LEN_HALF = 512;
constexpr uint32_t MAX_HEAD_NUM = 16;
constexpr uint32_t HEAD_ALIGN = 16;
constexpr uint32_t SEQ_PROC_SLICE = 16;
constexpr uint32_t CONST_TWO = 2;

uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

uint32_t Min(uint32_t a, uint32_t b)
{
    return a > b ? b : a;
}

template<typename T>
uint8_t *GetTilingBuf(T *tilingData)
{
    uint32_t tilingSize = tilingData->GetDataSize();
    uint8_t *buf = (uint8_t *)malloc(tilingSize);
    tilingData->SaveToBuffer(buf, tilingSize);
    return buf;
}


int GetPlatformInfo(const char *socVersion, BertSelfAttentionTilingData &tiling)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    if (ascendcPlatform == nullptr) {
        return -1;
    }

    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, tiling.l1BufferSize);
    return 0;
}


int GetSoftMaxTilingInfo(const char *socVersion, BertSelfAttentionTilingData &tiling)
{
    std::vector<int64_t> shapeVec = {tiling.seqProcSlice, tiling.seqLen};

    ge::Shape srcShape(shapeVec);
    optiling::SoftMaxTiling softMaxTiling;
    AscendC::SoftMaxTilingFunc(srcShape, sizeof(uint16_t), RESVER_SIZE, softMaxTiling);

    uint8_t *buf = GetTilingBuf<optiling::SoftMaxTiling>(&softMaxTiling);
    memcpy_s(reinterpret_cast<uint8_t *>(&tiling.softMaxTilingData),
        sizeof(tiling.softMaxTilingData), buf, sizeof(tiling.softMaxTilingData));
    return 0;
}


int ShapeCheck(BertSelfAttentionTilingData &tiling)
{
    if (tiling.seqLen > MAX_SEQ_LEN) {
        printf("seq:%d cant exceed max_seq_len:%d\n", tiling.seqLen, MAX_SEQ_LEN);
        return -1;
    }

    if ((tiling.seqLen % HEAD_ALIGN) != 0) {
        printf("seq:%d must mutiple of 16\n", tiling.seqLen);
        return -1;
    }

    if (tiling.headNum > MAX_HEAD_NUM) {
        printf("headNum:%d cant exceed max_head_num:%d\n", tiling.headNum, MAX_HEAD_NUM);
        return -1;
    }

    return 0;
}

int BroadCastTiling(const char *socVersion, BertSelfAttentionTilingData &tiling)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    std::vector<int64_t> srcShapeVec = {tiling.seqProcSlice, 1};
    std::vector<int64_t> dstShapeVec = {tiling.seqProcSlice, tiling.seqLen};

    ge::Shape srcShape(srcShapeVec);
    ge::Shape dstShape(dstShapeVec);

    uint32_t maxValue{0};
    uint32_t minValue{0};
    AscendC::GetBroadCastMaxMinTmpSize(*ascendcPlatform, srcShape, dstShape, CONST_TWO, false, maxValue, minValue);

    if (minValue >= RESVER_SIZE) {
        printf("min Value greater than 32K");
        return -1;
    }

    return 0;
}


int GenerateTiling(const char *socVersion, BertSelfAttentionTilingData &tiling)
{
    auto ret = ShapeCheck(tiling);
    if (ret == -1) {
        printf("ShapeCheck failed\n");
        return -1;
    }

    tiling.resverSize = RESVER_SIZE;
    tiling.seqProcSlice = SEQ_PROC_SLICE;
    if (tiling.seqProcSlice > MAX_SEQ_LEN_HALF) {
        tiling.seqProcSlice = SEQ_PROC_SLICE / CONST_TWO;
    }

    ret = GetPlatformInfo(socVersion, tiling);
    if (ret == -1) {
        printf("GetPlatformInfo failed\n");
        return -1;
    }

    ret = GetSoftMaxTilingInfo(socVersion, tiling);
    if (ret == -1) {
        printf("GetSoftMaxTilingInfo failed\n");
        return -1;
    }

    ret = BroadCastTiling(socVersion, tiling);
    if (ret == -1) {
        printf("BroadCastTiling failed\n");
        return -1;
    }

    tiling.rsqrtHeadDim = 1 / sqrt(tiling.headDim);
    return 1;
}
