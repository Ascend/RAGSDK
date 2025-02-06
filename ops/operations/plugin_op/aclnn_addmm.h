/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef MXRAGEMBFRAMEWORK_ACLNN_ADDMM_H
#define MXRAGEMBFRAMEWORK_ACLNN_ADDMM_H
#include "acl_nn_operation.h"

namespace atb_speed {
    namespace common {
        class AclnnAddmm : public AclNNOperation {
        public:
            explicit AclnnAddmm(const std::string &name);

            ~AclnnAddmm() override;

            atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                   atb::SVector<atb::TensorDesc> &outTensorDescs) const override;

            uint32_t GetInputNum() const override;

            uint32_t GetOutputNum() const override;

        private:
            int CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;
            int SetAclNNWorkspaceExecutor() override;
            int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

            int CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack);
            int CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack);

            bool alphaValue = true;
            bool betaValue = true;
            aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_BOOL);
            aclScalar* beta = aclCreateScalar(&betaValue, ACL_BOOL);
        };
    }
}


#endif // MXRAGEMBFRAMEWORK_ACLNN_ADDMM_H
