/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_H
#include <string>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "atb/atb_infer.h"
#include "atb/operation.h"
#include "acl_nn_operation_cache.h"

namespace atb_speed {
namespace common {

class AclNNOperation : public atb::Operation {
public:
    explicit AclNNOperation(const std::string &opName);
    virtual ~AclNNOperation();
    std::string GetName() const override;
    atb::Status Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context) override;
    atb::Status Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                        atb::Context *context) override;
    void DestroyOperation() const;

protected:
    atb::Status CreateAclNNOpCache(const atb::VariantPack &variantPack);
    atb::Status UpdateAclNNOpCache(const atb::VariantPack &variantPack);
    virtual int CreateAclNNVariantPack(const atb::VariantPack &variantPack) = 0;
    virtual int SetAclNNWorkspaceExecutor() = 0;
    virtual int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) = 0;
    std::shared_ptr<AclNNOpCache> aclnnOpCache_ = nullptr;
    std::string opName_;
};
} // namespace common
} // namespace atb_speed
#endif