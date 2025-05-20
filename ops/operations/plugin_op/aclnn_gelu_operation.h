/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#ifndef ATB_SPEED_PLUGIN_ACLNN_GELU_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_GELU_OPERATION_H

#include "acl_nn_operation.h"


namespace atb_speed::common {

    /// A struct defines `aclnnGelu` and `aclnnGelvV2` operation parameter.
    struct AclNNGeluParam {
        /// Indicates the gelu approximation algorithm to use.
        ///
        /// -1: use `aclnnGelu` operation, and use Tanh approximation approach to calculate Gelu.
        /// 0: use `aclnnGelvV2` operation, and use Cumulative Distribution Function for Gaussian Distribution.
        /// 1: use `aclnnGelvV2` operation, and use Tanh approximation approach to calculate Gelu.
        int64_t geluApproximate = -1;
    };

    /// This class defines a matrix operation that applies the Gaussian Error Linear Units function.
    ///
    /// This class makes use of `aclnnGeluGetWorkspaceSize` and `aclnnGeluV2GetWorkspaceSize` from AscendCL API.
    ///
    /// Operation's Inputs: \n
    /// | Name   | Dtype                    | Shape     | \n
    /// |--------|--------------------------|-----------| \n
    /// | x      | float32/float16/bfloat16 | [-1,…,-1] | \n
    ///
    /// Operation's Outputs: \n
    /// | Name   | Dtype                    | Shape     | \n
    /// |--------|--------------------------|-----------| \n
    /// | output | float32/float16/bfloat16 | [-1,…,-1] | \n
    ///
    ///
    /// \endcode
    class GeluOperation : public AclNNOperation {
    public:
        explicit GeluOperation(const std::string &name, AclNNGeluParam param);
        ~GeluOperation() override;
        atb::Status InferShape(
            const atb::SVector<atb::TensorDesc> &inTensorDesc,
            atb::SVector<atb::TensorDesc> &outTensorDesc
        ) const override;
        [[nodiscard]] uint32_t GetInputNum() const override;
        [[nodiscard]] uint32_t GetOutputNum() const override;

    protected:
        int SetAclNNWorkspaceExecutor() override;
        int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
        atb::Status CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;
        atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack);
        atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack);
        virtual std::shared_ptr<AclNNTensor> CreateTensor(atb::Tensor atbTensor, size_t tensorIdx);

    private:
        AclNNGeluParam param_;
        std::string opName_;
    };
}  // namespace atb_speed::common

#endif  // ATB_SPEED_PLUGIN_ACLNN_GELU_OPERATION_H
