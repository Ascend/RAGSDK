/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef ATB_SPEED_UTILS_CONFIG_H
#define ATB_SPEED_UTILS_CONFIG_H
#include <string>
#include <set>

namespace atb_speed {
class Config {
public:
    Config();
    ~Config();
    static std::string GetSaveTensorDir();
    bool IsSaveTensor() const;
    void DisableSaveTensor();
    uint64_t GetSaveTensorMaxNum() const;
    bool IsConvertNCHWToND() const;
    bool IsSaveTensorForRunner(const std::string &runnerName) const;
    bool IsTorchTensorFormatCast() const;
    bool IsUseTilingCopyStream() const;
    bool IsLayerInternalTensorReuse() const;

private:
    static bool IsEnable(const char *env, bool enable = false);
    void InitSaveTensor();
    void InitSaveTensor(const char *env, std::set<std::string> &nameSet) const;

private:
    bool isSaveTensor_ = false;
    uint64_t saveTensorMaxNum_ = 1;
    bool isConvertNCHWToND_ = false;
    bool isTorchTensorFormatCast_ = true;
    bool isUseTilingCopyStream_ = false;
    std::set<std::string> saveTensorRunnerNameSet_;
    bool isLayerInternalTensorReuse_ = false;
};
} // namespace atb_speed
#endif