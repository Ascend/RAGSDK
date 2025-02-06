/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#ifndef ATB_SPEED_PROBE_H
#define ATB_SPEED_PROBE_H

#include <string>
#include <iostream>

namespace atb_speed {

class SpeedProbe {
public:
    static bool IsReportModelTopoInfo(const std::string &modelName);
    static void ReportModelTopoInfo(const std::string &modelName, const std::string &graph);
};

}

#endif