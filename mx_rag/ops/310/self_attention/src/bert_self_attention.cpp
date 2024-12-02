/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */
#include "bert_self_attention_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr uint32_t MAX_HEAD_NUM = 16;

using namespace matmul;

enum class LinearParamType : uint8_t {
    QUERY = 0,
    KEY = 1,
    VALUE = 2,
};


const uint32_t CONST_TWO = 2;


__aicore__ inline uint32_t Min(uint32_t a, uint32_t b)
{
    return a > b ? b : a;
}

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline void CopyTiling(BertSelfAttentionTilingData *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(BertSelfAttentionTilingData) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

template <typename DataType> class BertSelfAttention {
public:
    __aicore__ inline BertSelfAttention(const BertSelfAttentionTilingData &tiling)
    {
        rsqrtHeadDim = static_cast<DataType>(tiling.rsqrtHeadDim);

        batchSize = tiling.batchSize;
        seqLen = tiling.seqLen;
        headNum = tiling.headNum;
        resverSize = tiling.resverSize;
        seqProcSlice = tiling.seqProcSlice;

        seq_process_size = seqProcSlice * seqLen;
        one_batch_size = headNum * seqLen * seqLen;
        one_head_size = seqLen * seqLen;

        softMaxTilingData = tiling.softMaxTilingData;
    }

    __aicore__ inline void BufferInit()
    {
        // ub total 256 * 1024
        pipe.InitBuffer(attention_scores_queue, BUFFER_NUM, seq_process_size * sizeof(DataType)); // 32K * 2 = 64K
        pipe.InitBuffer(attention_probs_queue, BUFFER_NUM, seq_process_size * sizeof(DataType)); // 32K * 2 = 64K

        pipe.InitBuffer(attention_mask_brd_queue, 1, seq_process_size * sizeof(DataType)); // 32 * 512 * 2 = 32K

        pipe.InitBuffer(attention_mask_queue, 1, seqLen * sizeof(DataType)); // 1K
        pipe.InitBuffer(ub_reserve_queue, 1, resverSize * sizeof(uint8_t)); // 90K

        reserve_tensor = ub_reserve_queue.AllocTensor<uint8_t>();
    }

    __aicore__ inline void Init(
        GM_ADDR attention_scores,
        GM_ADDR head_mask,
        GM_ADDR attention_mask,
        GM_ADDR attention_probs
    )
    {
        size_t input_size = batchSize * one_batch_size;
        size_t output_size = input_size;
        size_t attention_mask_size = batchSize * seqLen;
        size_t head_mask_size = headNum;

        attention_scores_gm.SetGlobalBuffer((__gm__ DataType *)attention_scores, input_size);
        head_mask_gm.SetGlobalBuffer((__gm__ DataType *)head_mask, head_mask_size);
        attention_mask_gm.SetGlobalBuffer((__gm__ DataType *)attention_mask, attention_mask_size);
        attention_probs_gm.SetGlobalBuffer((__gm__ DataType *)attention_probs, output_size);

        for (auto i = 0; i < headNum; i++) {
            head_mask_flag[i] = static_cast<float>(head_mask_gm.GetValue(i));
        }

        BufferInit();
    }

    __aicore__ inline void ProcessOneHead(
        uint32_t batch_idx, uint32_t head_idx, uint32_t seq_loop_times, uint32_t seq_tail_len)
    {
        const float float_zero = float(0.0);

        if (head_mask_flag[head_idx] == float_zero) {
            for (uint32_t seq_loop_idx = 0; seq_loop_idx < seq_loop_times - 1; seq_loop_idx++) {
                ZerosOutput(seqProcSlice);
                CopyOutOutput(batch_idx, head_idx, seq_loop_idx, seqProcSlice);
            }

            if (seq_tail_len != 0) {
                ZerosOutput(seq_tail_len);
                CopyOutOutput(batch_idx, head_idx, seq_loop_times - 1, seq_tail_len);
            }

            return;
        }

        for (uint32_t seq_loop_idx = 0; seq_loop_idx < seq_loop_times - 1; seq_loop_idx++) {
            CopyAttentionScoreFromL1(batch_idx, head_idx, seq_loop_idx, seqProcSlice);
            AddAttentionMask(seqProcSlice, false);
            DoSoftMax(seqProcSlice, false);
            CopyOutOutput(batch_idx, head_idx, seq_loop_idx, seqProcSlice);
        }

        if (seq_tail_len != 0) {
            CopyAttentionScoreFromL1(batch_idx, head_idx, seq_loop_times - 1, seq_tail_len);
            AddAttentionMask(seq_tail_len, true);
            DoSoftMax(seq_tail_len, true);
            CopyOutOutput(batch_idx, head_idx, seq_loop_times - 1, seq_tail_len);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t seq_loop_times = Ceiling(seqLen, seqProcSlice);
        uint32_t seq_tail_len = seqLen - (seq_loop_times - 1) * seqProcSlice;

        uint32_t each_core_proc_head_num = headNum / AscendC::GetBlockNum();
        uint32_t start_head_idx = AscendC::GetBlockIdx() * each_core_proc_head_num;

        // batchSize
        for (uint32_t batch_idx = 0; batch_idx < batchSize; batch_idx++) {
            BroadCastAttentionMask(batch_idx);

            for (uint32_t head_idx = start_head_idx;
                head_idx < start_head_idx + each_core_proc_head_num; head_idx++) {
                ProcessOneHead(batch_idx, head_idx, seq_loop_times, seq_tail_len);
            }

            FreeBrdCastAttentionMask();
        }

        ub_reserve_queue.FreeTensor<uint8_t>(reserve_tensor);
    }

private:
    __aicore__ inline void FreeBrdCastAttentionMask()
    {
        attention_mask_brd_queue.FreeTensor<DataType>(attention_mask_brd_tensor);
    }

    __aicore__ inline void CopyAttentionScoreFromL1(uint32_t batch_idx, uint32_t head_idx,
        uint32_t seq_idx, uint32_t seq_proc_size)
    {
        attention_scores_tensor = attention_scores_queue.AllocTensor<DataType>();

        int64_t offset = batch_idx * one_batch_size + head_idx * one_head_size
            + seq_idx * seq_process_size;

        AscendC::DataCopy(attention_scores_tensor, attention_scores_gm[offset], seq_proc_size * seqLen);

        attention_scores_queue.EnQue<DataType>(attention_scores_tensor);
        attention_scores_queue.DeQue<DataType>();
    }

    __aicore__ inline void AddAttentionMask(uint32_t seq_proc_size, bool is_tail)
    {
        AscendC::Muls(attention_scores_tensor, attention_scores_tensor, rsqrtHeadDim, seq_process_size);
        AscendC::Add(attention_scores_tensor, attention_scores_tensor, attention_mask_brd_tensor, seq_process_size);
    }

    __aicore__ inline void BroadCastAttentionMask(uint32_t batch_idx)
    {
        auto attention_mask_tensor = attention_mask_queue.AllocTensor<DataType>();
        attention_mask_brd_tensor = attention_mask_brd_queue.AllocTensor<DataType>();

        int64_t offset = batch_idx * seqLen;
        AscendC::DataCopy(attention_mask_tensor, attention_mask_gm[offset], seqLen);
        attention_mask_queue.EnQue<DataType>(attention_mask_tensor);
        attention_mask_queue.DeQue<DataType>();

        uint32_t dst_shape[CONST_TWO] = {seqProcSlice, seqLen};
        uint32_t src_shape[CONST_TWO] = {1, seqLen};
        AscendC::BroadCast<half, CONST_TWO, 0>(attention_mask_brd_tensor, attention_mask_tensor,
            dst_shape, src_shape, reserve_tensor);

        attention_mask_brd_queue.EnQue<DataType>(attention_mask_brd_tensor);
        attention_mask_brd_queue.DeQue<DataType>();

        attention_mask_queue.FreeTensor<DataType>(attention_mask_tensor);
    }

    __aicore__ inline void DoSoftMax(uint32_t seq_proc_size, bool is_tail)
    {
        auto attention_probs_tensor = attention_probs_queue.AllocTensor<DataType>();

        AscendC::SoftMaxShapeInfo srcShape = {seq_proc_size, seqLen, seq_proc_size, seqLen};
        AscendC::SoftMax<DataType, false, true>(attention_probs_tensor,
            attention_scores_tensor, reserve_tensor, softMaxTilingData, srcShape);

        attention_scores_queue.FreeTensor<DataType>(attention_scores_tensor);
        attention_probs_queue.EnQue<DataType>(attention_probs_tensor);
    }

    __aicore__ inline void ZerosOutput(uint32_t seq_proc_size)
    {
        auto attention_probs_tensor = attention_probs_queue.AllocTensor<DataType>();
        const half zeros = half(0.0);

        Duplicate(attention_probs_tensor, zeros, seq_process_size);

        attention_probs_queue.EnQue<DataType>(attention_probs_tensor);
    }

    __aicore__ inline void CopyOutOutput(uint32_t batch_idx, uint32_t head_idx,
        uint32_t seq_idx, uint32_t seq_proc_size)
    {
        auto attention_probs_tensor = attention_probs_queue.DeQue<DataType>();

        int64_t offset = batch_idx * one_batch_size + head_idx * one_head_size
            + seq_idx * seq_process_size;

        AscendC::DataCopy(attention_probs_gm[offset], attention_probs_tensor, seq_proc_size * seqLen);

        attention_probs_queue.FreeTensor<DataType>(attention_probs_tensor);
    }

private:
    AscendC::TPipe pipe;

    AscendC::GlobalTensor<DataType> attention_scores_gm;
    AscendC::GlobalTensor<DataType> head_mask_gm;
    AscendC::GlobalTensor<DataType> attention_mask_gm;
    AscendC::GlobalTensor<DataType> attention_probs_gm;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> attention_scores_queue; // 64K UB
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> attention_probs_queue; // 64K UB

    AscendC::TQue<AscendC::QuePosition::VECCALC, 1> attention_mask_brd_queue; // 64K UB

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> attention_mask_queue; // 128 UB
    AscendC::TQue<AscendC::QuePosition::VECCALC, 1> ub_reserve_queue; // 32K

    AscendC::LocalTensor<uint8_t> reserve_tensor;
    AscendC::LocalTensor<DataType> attention_mask_tensor;
    AscendC::LocalTensor<DataType> attention_mask_brd_tensor;
    AscendC::LocalTensor<DataType> attention_scores_tensor;
    BertSelfAttentionTilingData tilingData;

    DataType rsqrtHeadDim;

    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t headNum;
    uint32_t resverSize;
    uint32_t seqProcSlice;

    uint32_t seq_process_size;
    int64_t one_batch_size;
    int64_t one_head_size;

    float head_mask_flag[MAX_HEAD_NUM];

    SoftMaxTiling softMaxTilingData;
};

extern "C" __global__ __aicore__ void bert_self_attention(
    GM_ADDR attention_scores,
    GM_ADDR head_mask,
    GM_ADDR attention_mask,
    GM_ADDR attention_probs,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    BertSelfAttentionTilingData tilingData;
    CopyTiling(&tilingData, tiling);

    AscendC::SyncAll();


    SetSysWorkSpacePtr(workspace);
    BertSelfAttention<half> op(tilingData);

    op.Init(attention_scores,
            head_mask,
            attention_mask,
            attention_probs);

    op.Process();
}
