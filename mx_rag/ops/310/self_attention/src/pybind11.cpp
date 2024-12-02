/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <unistd.h>
#include "acl/acl.h"
#include "aclrtlaunch_bert_self_attention.h"
#include "bert_self_attention_tiling.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef CHECK_ACL
#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0)
#endif

const uint32_t CONST_TWO = 2;
const uint32_t CONST_ZERO = 0;
const uint32_t CONST_ONE = 1;

namespace mxrag
{
    class bert_self_attention {
    public:
        bert_self_attention() {}

        ~bert_self_attention() {
            if (tilingPtrHost != nullptr) {
                CHECK_ACL(aclrtFreeHost(tilingPtrHost));
            }

            if (tilingPtrDevice != nullptr) {
                CHECK_ACL(aclrtFree(tilingPtrDevice));
            }
        }

        at::Tensor exec(
            const at::Tensor &attention_scores,
            const at::Tensor &head_mask,
            const at::Tensor &attention_mask,
            uint32_t headDim)
        {
            auto acl_stream = c10_npu::getCurrentNPUStream();

            at::Tensor output = at::empty_like(attention_scores);
            auto ret = caculate_tiling(attention_scores, headDim);
            if (ret == -1) {
                printf("tiling failed\n");
                return output;
            }

            auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
            size_t usrWorkSpaceSize = 0;
            size_t sysWorkSpaceSize = static_cast<size_t>(ascendc_platform->GetLibApiWorkSpaceSize());
            size_t workSpaceSisze = usrWorkSpaceSize + sysWorkSpaceSize;
            auto workspace_tensor = at::empty({workSpaceSisze},
                at::TensorOptions().dtype(at::kByte).device(attention_scores.options().device()));

            uint32_t blockDim = 8;
        
            ACLRT_LAUNCH_KERNEL(bert_self_attention)
            (
                blockDim,
                acl_stream,
                const_cast<void *>(attention_scores.storage().data()),
                const_cast<void *>(head_mask.storage().data()),
                const_cast<void *>(attention_mask.storage().data()),
                const_cast<void *>(output.storage().data()),
                const_cast<void *>(workspace_tensor.storage().data()),
                reinterpret_cast<void *>(tilingPtrDevice)
            );

            return output;
        }

    private:
        bool need_caculate_tiling(const at::Tensor &attention_scores, uint32_t headDim)
        {
            if (!tilingInit) {
                CHECK_ACL(aclrtMallocHost((void **)(&tilingPtrHost),
                    sizeof(BertSelfAttentionTilingData)));
                CHECK_ACL(aclrtMalloc((void **)(&tilingPtrDevice),
                    sizeof(BertSelfAttentionTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
                tilingInit = true;
                return true;
            }

            uint32_t batchSize = attention_scores.size(0);
            uint32_t seqLen = attention_scores.size(2);
            uint32_t headNum = attention_scores.size(1);
            if (batchSize == cache_tiling.batchSize && seqLen == cache_tiling.seqLen &&
                headNum == cache_tiling.headNum && headDim == cache_tiling.headDim) {
                return false;
            }

            return true;
        }

        int caculate_tiling(const at::Tensor &attention_scores, uint32_t headDim)
        {
            if (!need_caculate_tiling(attention_scores, headDim)) {
                return 0;
            }

            BertSelfAttentionTilingData tiling;
            tiling.batchSize = attention_scores.size(CONST_ZERO);
            tiling.seqLen = attention_scores.size(CONST_TWO);
            tiling.headNum = attention_scores.size(CONST_ONE);
            tiling.headDim = headDim;

            auto ret = GenerateTiling("Ascend310P3", tiling);
            if (ret == -1) {
                printf("tiling failed\n");
                return -1;
            }

            CHECK_ACL(aclrtMemcpy(tilingPtrHost,
                sizeof(BertSelfAttentionTilingData),
                &tiling,
                sizeof(BertSelfAttentionTilingData),
                ACL_MEMCPY_HOST_TO_HOST));

            CHECK_ACL(aclrtMemcpy(tilingPtrDevice,
                sizeof(BertSelfAttentionTilingData),
                tilingPtrHost,
                sizeof(BertSelfAttentionTilingData),
                ACL_MEMCPY_HOST_TO_DEVICE));

            cache_tiling = tiling;
            return 0;
        }

    private:
        BertSelfAttentionTilingData *tilingPtrHost {nullptr};
        BertSelfAttentionTilingData *tilingPtrDevice {nullptr};

        BertSelfAttentionTilingData cache_tiling;

        bool tilingInit {false};
    };
}


PYBIND11_MODULE(mx_rag_opp, m)
{
    pybind11::class_<mxrag::bert_self_attention>(m, "bert_self_attention")
        .def(pybind11::init<>())
        .def("exec", &mxrag::bert_self_attention::exec);
}
