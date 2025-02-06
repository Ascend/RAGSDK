/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_MODEL_CREATOR_H
#define ATB_SPEED_MODEL_CREATOR_H
#define MODEL_CREATOR(ModelName, ModelParam)                                                                           \
    template <> Status CreateModel(const ModelParam &modelParam, Model **model)                                        \
    {                                                                                                                  \
        if (model == nullptr) {                                                                                        \
            return atb::ERROR_INVALID_PARAM;                                                                           \
        }                                                                                                              \
        *model = new ModelName(modelParam);                                                                            \
        return 0;                                                                                                      \
    }
#endif