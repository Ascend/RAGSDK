/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#ifndef ATB_SPEED_CONTEXT_FACTORY_H
#define ATB_SPEED_CONTEXT_FACTORY_H
#include <memory>
#include <atb/context.h>

namespace atb_speed {
class ContextFactory {
public:
    static std::shared_ptr<atb::Context> GetAtbContext(void *stream);
    static void FreeAtbContext();
};
}
#endif