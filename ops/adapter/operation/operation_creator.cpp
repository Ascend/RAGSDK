/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#include "operation_creator.h"

#include <functional>
#include <nlohmann/json.hpp>

#include "atb_speed/log.h"
#include "atb_speed/utils/operation_factory.h"


namespace atb_speed {

using OperationCreateFunc = std::function<atb::Operation *(const nlohmann::json &paramJson)>;

std::map<std::string, OperationCreateFunc> g_funcMap = {

};

atb::Operation *CreateOperation(const std::string &opName, const std::string &param)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) <<"parse param fail, please check param's format,error: " << e.what() << param;
        return nullptr;
    }

    auto operation = atb_speed::OperationFactory::CreateOperation(opName, paramJson);
    if (operation != nullptr) {
        ATB_LOG(INFO) << "Get Op from the OperationFactory, opName: " << opName;
        return operation;
    }

    auto it = g_funcMap.find(opName);
    if (it == g_funcMap.end()) {
        ATB_LOG(ERROR) << "not support opName:" << opName;
        return nullptr;
    }

    try {
        return it->second(paramJson);
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << opName << " parse json fail, error:" << e.what();
    }
    return nullptr;
}
} /* namespace atb */