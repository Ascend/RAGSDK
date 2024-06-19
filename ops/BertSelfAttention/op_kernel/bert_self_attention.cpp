/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: BertSelfAttention融合算子的kernel侧实现
 * Author: s00667951
 * Create: 2024-06-05
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

template <typename DataType>
class BertSelfAttention {
public:
    __aicore__ inline BertSelfAttention(){};

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR queryW, GM_ADDR queryBias, GM_ADDR keyW, GM_ADDR keyBias,
                                GM_ADDR valueW, GM_ADDR valueBias, GM_ADDR attentionMask, GM_ADDR dropOutMask,
                                GM_ADDR headMask, GM_ADDR output, GM_ADDR usrWorkspace,
                                const BertSelfAttentionTilingData *__restrict tiling);

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitBasic(const BertSelfAttentionTilingData *__restrict tiling);

    __aicore__ inline void InitBuffers();

    __aicore__ inline void ProcessProjections();

    __aicore__ inline void ProcessAttScore();

    __aicore__ inline void ProcessVectors();

    __aicore__ inline void CopyIn(uint64_t offset);

    __aicore__ inline void MulScalarCompute();

    __aicore__ inline void AddAttentionCompute(uint64_t batchId);

    __aicore__ inline void SoftMaxCompute();

    __aicore__ inline void DropOutCompute(uint64_t offset);

    __aicore__ inline void MulHeadCompute(uint64_t headId);

    __aicore__ inline void CopyInA(uint64_t offset);

    __aicore__ inline void CopyResOut(uint64_t resOffset);

    TPipe pipe;

    TCubeTiling projectionTiling;
    TCubeTiling attentionScoresTiling;
    TCubeTiling attentionProbsTiling;
    SoftMaxTiling softMaxTiling;

    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, DataType>, MatmulType<TPosition::GM, CubeFormat::ND, DataType>,
           MatmulType<TPosition::GM, CubeFormat::ND, DataType>, MatmulType<TPosition::GM, CubeFormat::ND, DataType> >
        projectionMatmulObj;

    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, DataType>,
           MatmulType<TPosition::GM, CubeFormat::ND, DataType, true>,
           MatmulType<TPosition::GM, CubeFormat::ND, DataType> >
        attentionScoresMatmulObj;

    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, DataType>, MatmulType<TPosition::GM, CubeFormat::ND, DataType>,
           MatmulType<TPosition::VECIN, CubeFormat::ND, DataType> >
        attentionProbsMatmulObj;

    TQue<QuePosition::VECIN, 1> inQueueVec;

    TQue<QuePosition::VECIN, 1> attentionMaskQueue;
    TQue<QuePosition::VECIN, 1> dropOutMaskQueue;
    TQue<QuePosition::VECIN, 1> softMaxWorkSpaceQueue;
    TQue<QuePosition::VECIN, 1> tmpQueue;

    GlobalTensor<DataType> inputGlobal;
    GlobalTensor<DataType> queryWGlobal;
    GlobalTensor<DataType> queryBiasGlobal;
    GlobalTensor<DataType> queryProjectedGlobal;
    GlobalTensor<DataType> queryTransedProjectedGlobal;
    GlobalTensor<DataType> keyWGlobal;
    GlobalTensor<DataType> keyBiasGlobal;
    GlobalTensor<DataType> keyProjectedGlobal;
    GlobalTensor<DataType> keyTransedProjectedGlobal;
    GlobalTensor<DataType> valueWGlobal;
    GlobalTensor<DataType> valueBiasGlobal;
    GlobalTensor<DataType> valueProjectedGlobal;
    GlobalTensor<DataType> valueTransedProjectedGlobal;
    GlobalTensor<DataType> attentionScoresGlobal;

    GlobalTensor<DataType> attentionMaskGlobal;
    GlobalTensor<uint8_t> dropOutMaskGlobal;
    GlobalTensor<DataType> headMaskGlobal;
    GlobalTensor<DataType> outputGlobal;

    LocalTensor<DataType> transUb;
    LocalTensor<DataType> invecLocal;
    LocalTensor<DataType> inputValueLocal;
    LocalTensor<DataType> outputLocal;

    LocalTensor<DataType> attentionMaskLocal;
    LocalTensor<uint8_t> dropOutMaskLocal;
    LocalTensor<uint8_t> softMaxWorkSpaceLocal;

    // split data by tiling info from MultiCoreMatmulTiling
    uint64_t mmAOffset;
    uint64_t mmBOffset;
    uint64_t mmCOffset;
    uint64_t mmBiasOffset;
    uint64_t mmASize;
    uint64_t mmBSize;

    uint32_t batchSize;
    uint32_t headNum;
    uint32_t aicNum;
    uint32_t aivNum;
    uint32_t usedAivCoreNum;
    uint32_t sequenceLength;
    uint32_t featuresSize;
    uint32_t headSize;
    uint32_t fomerBatchNum;
    uint32_t perBlockBatchFormer;
    uint32_t perBlockBatchLatter;
    uint32_t projectedVectorsMatSize;
    uint32_t attentionScoresMatSize;
    uint32_t projectedVectorsOffset;
    uint32_t attentionScoresOffset;
    uint32_t batchIdOffset;
    bool isNoTask{false};

    uint32_t softMaxWorkSpaceSize;

    uint32_t baseSeqLength;
    uint32_t tailSeqLength;
    uint32_t computeSeqLength;

    DataType recSqrtHeadSize;
    float mulProbValue;

    uint32_t formerNum;
    uint32_t probePerBlockFormer;
    uint32_t probePerBlockLatter;
    uint32_t probePerBlock;
    uint32_t beginProbeId;

    uint32_t blkIdx;
};

template <typename DataType>
__aicore__ inline void BertSelfAttention<DataType>::InitBasic(const BertSelfAttentionTilingData *__restrict tiling)
{
    this->blkIdx = GetBlockIdx();
    this->projectionTiling = tiling->projectionTiling;
    this->attentionScoresTiling = tiling->scoresCubeTiling;
    this->attentionProbsTiling = tiling->probsCubeTiling;
    this->projectedVectorsMatSize = attentionScoresTiling.M * attentionScoresTiling.Ka;  // 左右矩阵等大小
    this->attentionScoresMatSize = attentionScoresTiling.M * attentionScoresTiling.N;
    this->batchSize = tiling->batchSize;
    this->headNum = tiling->headNum;
    this->aicNum = tiling->aicNum;
    this->aivNum = tiling->aivNum;
    this->usedAivCoreNum = tiling->usedAivCoreNum;
    this->sequenceLength = tiling->sequenceLength;
    this->featuresSize = tiling->featuresSize;
    this->headSize = tiling->headSize;
    this->fomerBatchNum = tiling->fomerBatchNum;
    this->perBlockBatchFormer = tiling->perBlockBatchFormer;
    this->perBlockBatchLatter = tiling->perBlockBatchLatter;
    this->isNoTask = (blkIdx >= usedAivCoreNum);
    if (blkIdx <= fomerBatchNum) {
        this->projectedVectorsOffset = blkIdx * perBlockBatchFormer * projectedVectorsMatSize;
        this->attentionScoresOffset = blkIdx * perBlockBatchFormer * attentionScoresMatSize;
        this->batchIdOffset = blkIdx * perBlockBatchFormer;
    } else {
        this->projectedVectorsOffset = fomerBatchNum * perBlockBatchFormer * projectedVectorsMatSize +
                                       (blkIdx - fomerBatchNum) * perBlockBatchLatter * projectedVectorsMatSize;
        this->attentionScoresOffset = fomerBatchNum * perBlockBatchFormer * attentionScoresMatSize +
                                      (blkIdx - fomerBatchNum) * perBlockBatchLatter * attentionScoresMatSize;
        this->batchIdOffset = fomerBatchNum * perBlockBatchFormer + (blkIdx - fomerBatchNum) * perBlockBatchLatter;
    }
    this->softMaxTiling = tiling->softMaxTilingData;
    this->softMaxWorkSpaceSize = tiling->softMaxWorkSpaceSize;
    this->baseSeqLength = tiling->baseSeqLength;
    this->tailSeqLength = tiling->tailSeqLength;
    this->recSqrtHeadSize = static_cast<DataType>(1.0f / tiling->sqrtHeadSize);
    this->mulProbValue = 1.0f - tiling->dropOutKeepProb;
    // 单个query需要由当前核处理的probe数量
    uint32_t partNum = batchSize * headNum * ((sequenceLength + baseSeqLength - 1) / baseSeqLength);
    this->formerNum = partNum % aivNum;
    this->probePerBlockFormer = (partNum + aivNum - 1) / aivNum;
    this->probePerBlockLatter = partNum / aivNum;
    this->probePerBlock = (blkIdx < formerNum ? probePerBlockFormer : probePerBlockLatter);
    this->beginProbeId = (blkIdx < formerNum
                                ? (probePerBlockFormer * blkIdx)
                                : (probePerBlockFormer * formerNum + (blkIdx - formerNum) * probePerBlockLatter));
}

template <typename DataType>
__aicore__ inline void BertSelfAttention<DataType>::InitBuffers()
{
    if (baseSeqLength < headSize) {
        pipe.InitBuffer(inQueueVec, 1, headSize * sequenceLength * sizeof(DataType));
    } else {
        pipe.InitBuffer(inQueueVec, 1, baseSeqLength * sequenceLength * sizeof(DataType));
    }
    pipe.InitBuffer(attentionMaskQueue, 1, sequenceLength * sizeof(DataType));
    pipe.InitBuffer(dropOutMaskQueue, 1,
                    baseSeqLength * sequenceLength * sizeof(uint8_t) / 8);  // sequenceLength>=16，且为16的倍数
    pipe.InitBuffer(softMaxWorkSpaceQueue, 1, softMaxWorkSpaceSize);
    pipe.InitBuffer(tmpQueue, 1, 1024);
}

template <typename DataType>
__aicore__ inline void BertSelfAttention<DataType>::Init(GM_ADDR input, GM_ADDR queryW, GM_ADDR queryBias, GM_ADDR keyW,
                                                         GM_ADDR keyBias, GM_ADDR valueW, GM_ADDR valueBias,
                                                         GM_ADDR attentionMask, GM_ADDR dropOutMask, GM_ADDR headMask,
                                                         GM_ADDR output, GM_ADDR usrWorkspace,
                                                         const BertSelfAttentionTilingData *__restrict tiling)
{
    InitBasic(tiling);
    inputGlobal.SetGlobalBuffer((__gm__ DataType *)input);
    queryWGlobal.SetGlobalBuffer((__gm__ DataType *)queryW);
    queryBiasGlobal.SetGlobalBuffer((__gm__ DataType *)queryBias);
    keyWGlobal.SetGlobalBuffer((__gm__ DataType *)keyW);
    keyBiasGlobal.SetGlobalBuffer((__gm__ DataType *)keyBias);
    valueWGlobal.SetGlobalBuffer((__gm__ DataType *)valueW);
    valueBiasGlobal.SetGlobalBuffer((__gm__ DataType *)valueBias);
    outputGlobal.SetGlobalBuffer((__gm__ DataType *)output);
    queryProjectedGlobal.SetGlobalBuffer((__gm__ DataType *)usrWorkspace);
    keyProjectedGlobal.SetGlobalBuffer((__gm__ DataType *)usrWorkspace + batchSize * sequenceLength * featuresSize);
    valueProjectedGlobal.SetGlobalBuffer((__gm__ DataType *)usrWorkspace +
                                         2 * batchSize * sequenceLength * featuresSize);
    queryTransedProjectedGlobal.SetGlobalBuffer((__gm__ DataType *)usrWorkspace +
                                                3 * batchSize * sequenceLength * featuresSize);
    keyTransedProjectedGlobal.SetGlobalBuffer((__gm__ DataType *)usrWorkspace +
                                              4 * batchSize * sequenceLength * featuresSize);
    valueTransedProjectedGlobal.SetGlobalBuffer((__gm__ DataType *)usrWorkspace +
                                                5 * batchSize * sequenceLength * featuresSize);
    attentionScoresGlobal.SetGlobalBuffer((__gm__ DataType *)usrWorkspace +
                                          6 * batchSize * sequenceLength * featuresSize);
    attentionMaskGlobal.SetGlobalBuffer((__gm__ DataType *)attentionMask);
    dropOutMaskGlobal.SetGlobalBuffer((__gm__ uint8_t *)dropOutMask);
    headMaskGlobal.SetGlobalBuffer((__gm__ DataType *)headMask);
    auto nNum = Ceil(projectionTiling.N, projectionTiling.singleCoreN);
    auto mNum = Ceil(projectionTiling.M, projectionTiling.singleCoreM);
    auto coreNId = blkIdx % nNum;
    auto coreMId = blkIdx / nNum;
    mmAOffset = coreMId * projectionTiling.singleCoreM * projectionTiling.Ka;
    mmBOffset = coreNId * projectionTiling.singleCoreN;
    mmCOffset = coreMId * projectionTiling.singleCoreM * projectionTiling.N + coreNId * projectionTiling.singleCoreN;
    mmBiasOffset = coreNId * projectionTiling.singleCoreN;
    mmASize = (coreMId != (mNum - 1)) ? projectionTiling.singleCoreM
                                        : (projectionTiling.M - coreMId * (projectionTiling.singleCoreM));
    mmBSize = (coreNId != (nNum - 1)) ? projectionTiling.singleCoreN
                                        : (projectionTiling.N - coreNId * (projectionTiling.singleCoreN));
    InitBuffers();
}

template <typename DataType>
__aicore__ inline void BertSelfAttention<DataType>::ProcessProjections()
{
    projectionMatmulObj.SetTensorA(inputGlobal[mmAOffset]);
    projectionMatmulObj.SetTensorB(queryWGlobal[mmBOffset]);
    projectionMatmulObj.SetBias(queryBiasGlobal[mmBiasOffset]);
    projectionMatmulObj.SetTail(mmASize, mmBSize, projectionTiling.Ka);
    projectionMatmulObj.IterateAll(queryProjectedGlobal[mmCOffset]);
    projectionMatmulObj.End();
    SyncAll();

    projectionMatmulObj.SetTensorA(inputGlobal[mmAOffset]);
    projectionMatmulObj.SetTensorB(keyWGlobal[mmBOffset]);
    projectionMatmulObj.SetBias(keyBiasGlobal[mmBiasOffset]);
    projectionMatmulObj.SetTail(mmASize, mmBSize, projectionTiling.Ka);
    projectionMatmulObj.IterateAll(keyProjectedGlobal[mmCOffset]);
    projectionMatmulObj.End();
    SyncAll();

    projectionMatmulObj.SetTensorA(inputGlobal[mmAOffset]);
    projectionMatmulObj.SetTensorB(valueWGlobal[mmBOffset]);
    projectionMatmulObj.SetBias(valueBiasGlobal[mmBiasOffset]);
    projectionMatmulObj.SetTail(mmASize, mmBSize, projectionTiling.Ka);
    projectionMatmulObj.IterateAll(valueProjectedGlobal[mmCOffset]);
    projectionMatmulObj.End();
    SyncAll();
}

template <typename DataType>
__aicore__ inline void BertSelfAttention<DataType>::ProcessAttScore()
{
    if (!isNoTask) {
        int curHeadNum = (blkIdx < fomerBatchNum) ? perBlockBatchFormer : perBlockBatchLatter;
        transUb = inQueueVec.AllocTensor<DataType>();
        for (int transLoopIds = 0; transLoopIds < curHeadNum; ++transLoopIds) {
            pipe_barrier(PIPE_ALL);  // useful
            auto transBatchId = (batchIdOffset + transLoopIds) / headNum;
            auto transHeadId = (batchIdOffset + transLoopIds) % headNum;
            auto transOffsetCopyIn = transBatchId * (sequenceLength * featuresSize) + transHeadId * headSize;
            auto transOffsetCopyOut = transBatchId * (sequenceLength * featuresSize) +
                                      transHeadId * sequenceLength * headSize;

            pipe_barrier(PIPE_ALL);  // useful
            DataCopy(transUb, queryProjectedGlobal[transOffsetCopyIn],
                     { static_cast<uint16_t>(sequenceLength), static_cast<uint16_t>(headSize * sizeof(DataType) / 32),
                       static_cast<uint16_t>((headNum - 1) * headSize * sizeof(DataType) / 32), 0 });
            pipe_barrier(PIPE_ALL);  // useful
            DataCopy(queryTransedProjectedGlobal[transOffsetCopyOut], transUb, sequenceLength * headSize);

            pipe_barrier(PIPE_ALL);  // useful
            DataCopy(transUb, keyProjectedGlobal[transOffsetCopyIn],
                     { static_cast<uint16_t>(sequenceLength), static_cast<uint16_t>(headSize * sizeof(DataType) / 32),
                       static_cast<uint16_t>((headNum - 1) * headSize * sizeof(DataType) / 32), 0 });
            pipe_barrier(PIPE_ALL);  // useful
            DataCopy(keyTransedProjectedGlobal[transOffsetCopyOut], transUb, sequenceLength * headSize);

            pipe_barrier(PIPE_ALL);  // useful
            DataCopy(transUb, valueProjectedGlobal[transOffsetCopyIn],
                     { static_cast<uint16_t>(sequenceLength), static_cast<uint16_t>(headSize * sizeof(DataType) / 32),
                       static_cast<uint16_t>((headNum - 1) * headSize * sizeof(DataType) / 32), 0 });
            pipe_barrier(PIPE_ALL);  // useful
            DataCopy(valueTransedProjectedGlobal[transOffsetCopyOut], transUb, sequenceLength * headSize);
        }

        inQueueVec.FreeTensor(transUb);
        pipe_barrier(PIPE_ALL);  // useful
        for (int headCnt = 0; headCnt < curHeadNum; ++headCnt) {
            attentionScoresMatmulObj.SetTensorA(
                queryTransedProjectedGlobal[projectedVectorsOffset + headCnt * projectedVectorsMatSize]);
            attentionScoresMatmulObj.SetTensorB(
                keyTransedProjectedGlobal[projectedVectorsOffset + headCnt * projectedVectorsMatSize], true);
            attentionScoresMatmulObj.IterateAll(
                attentionScoresGlobal[attentionScoresOffset + headCnt * attentionScoresMatSize]);
            attentionScoresMatmulObj.End();
        }
    }
    SyncAll();
}

template <typename DataType>
__aicore__ inline void BertSelfAttention<DataType>::ProcessVectors()
{
    for (uint64_t probeId = beginProbeId; probeId < beginProbeId + probePerBlock; probeId++) {
        pipe_barrier(PIPE_ALL);
        uint64_t sequencePart = (sequenceLength + baseSeqLength - 1) / baseSeqLength;
        uint64_t batchId = probeId / (headNum * sequencePart);
        uint64_t headId = (probeId % (headNum * sequencePart)) / sequencePart;
        uint64_t sequencePartId = (probeId % (headNum * sequencePart)) % sequencePart;
        this->computeSeqLength = (sequencePartId == sequencePart - 1) ? tailSeqLength : baseSeqLength;
        uint64_t offset = batchId * headNum * sequenceLength * sequenceLength +
                          headId * sequenceLength * sequenceLength + sequencePartId * baseSeqLength * sequenceLength;
        pipe_barrier(PIPE_ALL);
        CopyIn(offset);

        pipe_barrier(PIPE_ALL);
        MulScalarCompute();

        pipe_barrier(PIPE_ALL);
        AddAttentionCompute(batchId);

        pipe_barrier(PIPE_ALL);
        SoftMaxCompute();
        pipe_barrier(PIPE_ALL);  // useful
        DropOutCompute(offset);

        pipe_barrier(PIPE_ALL);
        MulHeadCompute(headId);

        pipe_barrier(PIPE_ALL);  // useful
        CopyInA(offset);

        pipe_barrier(PIPE_ALL);
        attentionProbsMatmulObj.SetTensorA(attentionScoresGlobal[offset]);

        uint64_t valueOffset = batchId * headNum * sequenceLength * headSize + headId * sequenceLength * headSize;
        uint64_t resOffset = batchId * headNum * sequenceLength * headSize + headId * headSize +
                             sequencePartId * baseSeqLength * headNum * headSize;
        outputLocal = inQueueVec.AllocTensor<DataType>();
        attentionProbsMatmulObj.SetTensorB(valueTransedProjectedGlobal[valueOffset]);
        attentionProbsMatmulObj.SetTail(computeSeqLength, headSize, sequenceLength);
        attentionProbsMatmulObj.template IterateAll<true>(outputLocal, 0);
        attentionProbsMatmulObj.End();
        pipe_barrier(PIPE_ALL);  // useful
        inQueueVec.EnQue(outputLocal);
        CopyResOut(resOffset);
    }
}

template <typename DataType>
__aicore__ inline void BertSelfAttention<DataType>::Process()
{
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), projectionMatmulObj, &projectionTiling, attentionScoresMatmulObj,
                      &attentionScoresTiling, attentionProbsMatmulObj, &attentionProbsTiling);

    ProcessProjections();

    ProcessAttScore();

    ProcessVectors();
}

template <typename DataType>
__aicore__ inline void BertSelfAttention<DataType>::CopyIn(uint64_t offset)
{
    invecLocal = inQueueVec.AllocTensor<DataType>();
    DataCopy(invecLocal, attentionScoresGlobal[offset], computeSeqLength * sequenceLength);
    inQueueVec.EnQue(invecLocal);
}

template <typename DataType>
__aicore__ inline void BertSelfAttention<DataType>::MulScalarCompute()
{
    invecLocal = inQueueVec.DeQue<DataType>();
    Muls(invecLocal, invecLocal, recSqrtHeadSize, computeSeqLength * sequenceLength);
    inQueueVec.EnQue(invecLocal);
}

template <typename DataType>
__aicore__ inline void BertSelfAttention<DataType>::AddAttentionCompute(uint64_t batchId)
{
    invecLocal = inQueueVec.DeQue<DataType>();
    attentionMaskLocal = attentionMaskQueue.AllocTensor<DataType>();
    DataCopy(attentionMaskLocal, attentionMaskGlobal[batchId * sequenceLength], sequenceLength);
    pipe_barrier(PIPE_ALL);  // useful
    for (int i = 0; i < computeSeqLength; i++) {
        uint64_t offset = i * sequenceLength;
        Add(invecLocal[offset], invecLocal[offset], attentionMaskLocal, sequenceLength);
    }
    attentionMaskQueue.FreeTensor(attentionMaskLocal);
    inQueueVec.EnQue(invecLocal);
}

template <typename DataType>
__aicore__ inline void BertSelfAttention<DataType>::SoftMaxCompute()
{
    invecLocal = inQueueVec.DeQue<DataType>();
    softMaxWorkSpaceLocal = softMaxWorkSpaceQueue.AllocTensor<uint8_t>();
    SoftMax(invecLocal, invecLocal, softMaxWorkSpaceLocal, softMaxTiling,
            { computeSeqLength, sequenceLength, computeSeqLength, sequenceLength });
    softMaxWorkSpaceQueue.FreeTensor(softMaxWorkSpaceLocal);
    inQueueVec.EnQue(invecLocal);
}

template <typename DataType>
__aicore__ inline void BertSelfAttention<DataType>::DropOutCompute(uint64_t offset)
{
    const DataType zero = 0.0f;
    invecLocal = inQueueVec.DeQue<DataType>();
    if (mulProbValue == 0) {
        Duplicate(invecLocal, zero, computeSeqLength * sequenceLength);
    } else if (mulProbValue != 1) {
        const DataType tmpMulProbValue = 1 / mulProbValue;
        dropOutMaskLocal = dropOutMaskQueue.AllocTensor<uint8_t>();

        DataCopy(dropOutMaskLocal, dropOutMaskGlobal[offset / 8], computeSeqLength * sequenceLength / 8);  // 需要大于32B
        pipe_barrier(PIPE_ALL);                                                                            // useful

        Select(invecLocal, dropOutMaskLocal, invecLocal, zero, SELMODE::VSEL_TENSOR_SCALAR_MODE,
               computeSeqLength * sequenceLength);
        Muls(invecLocal, invecLocal, tmpMulProbValue, computeSeqLength * sequenceLength);

        dropOutMaskQueue.FreeTensor(dropOutMaskLocal);
    } else {
        // Do Noting
    }
    inQueueVec.EnQue(invecLocal);
}

template <typename DataType>
__aicore__ inline void BertSelfAttention<DataType>::MulHeadCompute(uint64_t headId)
{
    invecLocal = inQueueVec.DeQue<DataType>();
    const DataType headMaskValue = headMaskGlobal.GetValue(headId);
    Muls(invecLocal, invecLocal, headMaskValue, computeSeqLength * sequenceLength);
    inQueueVec.EnQue(invecLocal);
}

template <typename DataType>
__aicore__ inline void BertSelfAttention<DataType>::CopyInA(uint64_t offset)
{
    invecLocal = inQueueVec.DeQue<DataType>();
    DataCopy(attentionScoresGlobal[offset], invecLocal, computeSeqLength * sequenceLength);
    inQueueVec.FreeTensor(invecLocal);
}

template <typename DataType>
__aicore__ inline void BertSelfAttention<DataType>::CopyResOut(uint64_t resOffset)
{
    outputLocal = inQueueVec.DeQue<DataType>();
    DataCopy(outputGlobal[resOffset], outputLocal,
             { static_cast<uint16_t>(computeSeqLength), static_cast<uint16_t>(headSize * sizeof(DataType) / 32), 0,
               static_cast<uint16_t>((headNum - 1) * headSize * sizeof(DataType) / 32) });
    inQueueVec.FreeTensor(outputLocal);
}

extern "C" __global__ __aicore__ void bert_self_attention(GM_ADDR input, GM_ADDR queryW, GM_ADDR queryBias,
                                                          GM_ADDR keyW, GM_ADDR keyBias, GM_ADDR valueW,
                                                          GM_ADDR valueBias, GM_ADDR attentionMask, GM_ADDR dropOutMask,
                                                          GM_ADDR headMask, GM_ADDR output, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    GET_TILING_DATA(tiling_data, tiling);
    const BertSelfAttentionTilingData *__restrict tiling_device = &tiling_data;

    if (TILING_KEY_IS(0)) {
        BertSelfAttention<float> op;
        op.Init(input, queryW, queryBias, keyW, keyBias, valueW, valueBias, attentionMask, dropOutMask, headMask,
                output, usrWorkspace, tiling_device);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        BertSelfAttention<half> op;
        op.Init(input, queryW, queryBias, keyW, keyBias, valueW, valueBias, attentionMask, dropOutMask, headMask,
                output, usrWorkspace, tiling_device);
        op.Process();
    } else {
        return;
    }
}