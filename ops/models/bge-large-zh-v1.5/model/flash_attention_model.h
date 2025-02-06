/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_MODELS_BGE_LARGE_ZH_FLASH_ATTENTION_MODEL_H
#define ATB_SPEED_MODELS_BGE_LARGE_ZH_FLASH_ATTENTION_MODEL_H

#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace bge_large {
class FlashAttentionModel : public Model {
public:
    struct Param {
        double layerNormEps = 0;
        int headNum = 0;
        int dk = 0;
        int layerNum = 0;
        float qkScale = 1.0;
        int rank = 0;
        int rankSize = 1;
        void FromString(const std::string &param);
    };
    explicit FlashAttentionModel(const std::string &param);
    ~FlashAttentionModel();
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
        std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    int64_t BuildGraph() override;
    atb::Status BuildWordEmbeddingNode(int& nodeId);
    atb::Status BuildTokentypeEmbeddingNode(int& nodeId);
    atb::Status BuildPositionEmbeddingNode(int& nodeId);
    atb::Status BuildLayerNormNode(int& nodeId);
    atb::Status BuildPositionIdsNode(int& nodeId);
    atb::Status BuildAddNode(int& nodeId);
    void BuildLayerNode(int& nodeId);
    
    Param param_;
    atb::Status ParseParam(const std::string &param) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};
} // namespace bge_large
} // namespace atb_speed

#endif // BGE_LARGE_ZH_FLASH_ATTENTION_MODEL_H
