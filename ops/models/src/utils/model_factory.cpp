/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "atb_speed/utils/model_factory.h"
#include "atb_speed/log.h"

namespace atb_speed {
bool ModelFactory::Register(const std::string &modelName, CreateModelFuncPtr createModel)
{
    auto it = ModelFactory::GetRegistryMap().find(modelName);
    if (it != ModelFactory::GetRegistryMap().end()) {
        ATB_LOG(WARN) << modelName << " model already exists, but the duplication doesn't matter.";
        return false;
    }
    ModelFactory::GetRegistryMap()[modelName] = createModel;
    return true;
}

std::shared_ptr<atb_speed::Model> ModelFactory::CreateInstance(const std::string &modelName, const std::string &param)
{
    auto it = ModelFactory::GetRegistryMap().find(modelName);
    if (it != ModelFactory::GetRegistryMap().end()) {
        ATB_LOG(INFO) << "find model: " << modelName;
        return it->second(param);
    }
    ATB_LOG(WARN) << "ModelName: " << modelName << " not find in model factory map";
    return nullptr;
}

std::unordered_map<std::string, CreateModelFuncPtr> &ModelFactory::GetRegistryMap()
{
    static std::unordered_map<std::string, CreateModelFuncPtr> modelRegistryMap;
    return modelRegistryMap;
}
} // namespace atb_speed
