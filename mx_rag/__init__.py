#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import sys

from mx_rag.version import __version__

sys.tracebacklimit = 0

# 默认关闭ragas的track
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
# 默认HF_HUB离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
# 默认datasets离线模式
os.environ["HF_DATASETS_OFFLINE"] = "1"
