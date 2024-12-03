# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

__all__ = ["BertSelfAttentionSpeed", "set_self_attention_enable_status"]

from mx_rag.ops.atlas.self_attention.patch.bert_patch_310 import BertSelfAttentionSpeed
from mx_rag.ops.atlas.self_attention.patch.bert_patch_310 import set_self_attention_enable_status
