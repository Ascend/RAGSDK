/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_UTILS_SINGLETON_H
#define ATB_SPEED_UTILS_SINGLETON_H

namespace atb_speed {
template <class T> T &GetSingleton()
{
    static T instance;
    return instance;
}
} // namespace atb_speed
#endif