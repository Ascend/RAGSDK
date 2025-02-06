/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "atb_speed/utils/speed_probe.h"

namespace atb_speed {

bool SpeedProbe::IsReportModelTopoInfo(const std::string &modelName)
{
    (void)modelName;
    return false;
}

void SpeedProbe::ReportModelTopoInfo(const std::string &modelName, const std::string &graph)
{
    (void)modelName;
    (void)graph;
    return;
}

} // namespace atb_speed