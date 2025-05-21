/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "atb_speed/base/context_factory.h"
#include <thread>
#include <acl/acl.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/config.h"

namespace atb_speed {
thread_local std::shared_ptr<atb::Context> g_localContext;

std::shared_ptr<atb::Context> ContextFactory::GetAtbContext(aclrtStream stream)
{
    if (g_localContext) {
    ATB_LOG(INFO) << "ContextFactory return localContext";
    return g_localContext;
    }
    ATB_LOG(INFO) << "ContextFactory create atb::Context start";
    atb::Context *context = nullptr;
    atb::Status st = atb::CreateContext(&context);
    ATB_LOG_IF(st != 0, ERROR) << "ContextFactory create atb::Context fail";

    if (context) {
        context->SetExecuteStream(stream);
        if (atb_speed::GetSingleton<atb_speed::Config>().IsUseTilingCopyStream()) {
            ATB_LOG(INFO) << "ContextFactory use tiling copy stream";
            context->SetAsyncTilingCopyStatus(true);
        } else {
            ATB_LOG(INFO) << "ContextFactory not use tiling copy stream";
        }
    }

    std::shared_ptr<atb::Context> tmpLocalContext(context, [](atb::Context* context) {atb::DestroyContext(context);});
    g_localContext = tmpLocalContext;

    return g_localContext;
}

void ContextFactory::FreeAtbContext()
{
    ATB_LOG(INFO) << "ContextFactory FreeAtbContext start.";
    if (!g_localContext) {
        return;
    }
    
    ATB_LOG(INFO) << "ContextFactory localContext use_count: " << g_localContext.use_count();
    if (g_localContext.use_count() != 1) {
        return;
    }
    ATB_LOG(INFO) << "ContextFactory localContext reset.";
    g_localContext.reset();
}
}