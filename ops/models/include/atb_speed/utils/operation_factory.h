/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_UTILS_OPERATION_FACTORY_H
#define ATB_SPEED_UTILS_OPERATION_FACTORY_H

#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

#include "atb/operation.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"


namespace atb_speed {
using CreateOperationFuncPtr = std::function<atb::Operation *(const nlohmann::json &)>;

class OperationFactory {
public:
    static bool Register(const std::string &operationName, CreateOperationFuncPtr createOperation);
    static atb::Operation *CreateOperation(const std::string &operationName, const nlohmann::json &param);
private:
    static std::unordered_map<std::string, CreateOperationFuncPtr> &GetRegistryMap();
};

#define OPERATION_NAMESPACE_STRINGIFY(operationNameSpace) #operationNameSpace
#define REGISTER_OPERATION(nameSpace, operationCreateFunc)                                                   \
        struct Register##_##nameSpace##_##operationCreateFunc {                                              \
            inline Register##_##nameSpace##_##operationCreateFunc()                                          \
            {                                                                                                \
                ATB_LOG(INFO) << "register operation " << #nameSpace << "_" << #operationCreateFunc;                   \
                OperationFactory::Register(OPERATION_NAMESPACE_STRINGIFY(nameSpace##_##operationCreateFunc), \
                    &(operationCreateFunc));                                                                 \
            }                                                                                                \
        } static instance_##nameSpace##operationCreateFunc
} // namespace atb_speed
#endif