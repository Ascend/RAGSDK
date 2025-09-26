#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Description: SDK uninstallation tool.
# Author: Mind SDK
# Create: 2025
# History: NA

export RAG_SDK_HOME=/home/HwHiAiUser/Ascend
export PYTHONPATH=$RAG_SDK_HOME/ops/transformer_adapter:$PYTHONPATH
export LD_LIBRARY_PATH=$RAG_SDK_HOME/ops/lib/:$LD_LIBRARY_PATH