# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
__all__ = ["enable_bert_speed",
           "enable_roberta_speed",
           "enable_xlm_roberta_speed",
           "enable_clip_speed"]

from modeling_bert_adapter import enable_bert_speed
from modeling_roberta_adapter import enable_roberta_speed
from modeling_xlm_roberta_adapter import enable_xlm_roberta_speed
from modeling_clip_adapter import enable_clip_speed