/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_EXECUTOR_MANAGER_H
#define ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_EXECUTOR_MANAGER_H

#include <map>
#include <string>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>

namespace atb_speed {
namespace common {

class ExecutorManager {
public:
    int IncreaseReference(aclOpExecutor *executor);
    int DecreaseReference(aclOpExecutor *executor);
    int GetReference(aclOpExecutor *executor);
    std::string PrintExecutorCount();

private:
    std::map<aclOpExecutor *, int> executorCount_;
};

} // namespace common
} // namespace atb_speed
#endif