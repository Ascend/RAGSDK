/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_UTILS_MODEL_FACTORY_H
#define ATB_SPEED_UTILS_MODEL_FACTORY_H

#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

#include "atb_speed/base/model.h"
#include "atb_speed/log.h"

namespace atb_speed {
using CreateModelFuncPtr = std::function<std::shared_ptr<atb_speed::Model>(const std::string &)>;

class ModelFactory {
public:
    static bool Register(const std::string &modelName, CreateModelFuncPtr createModel);
    static std::shared_ptr<atb_speed::Model> CreateInstance(const std::string &modelName, const std::string &param);
private:
    static std::unordered_map<std::string, CreateModelFuncPtr> &GetRegistryMap();
};

#define MODEL_NAMESPACE_STRINGIFY(modelNameSpace) #modelNameSpace
#define REGISTER_MODEL(nameSpace, modelName)                                                      \
        struct Register##_##nameSpace##_##modelName {                                             \
            inline Register##_##nameSpace##_##modelName() noexcept                                \
            {                                                                                     \
                ATB_LOG(INFO) << "register model " << #nameSpace << "_" << #modelName;            \
                ModelFactory::Register(MODEL_NAMESPACE_STRINGIFY(nameSpace##_##modelName),        \
                    [](const std::string &param) { return std::make_shared<modelName>(param); }); \
            }                                                                                     \
        } static instance_##nameSpace##modelName
} // namespace atb_speed
#endif