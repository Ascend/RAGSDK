#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

RAGSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
import math
import os
from typing import Tuple, Optional

import torch
import torch_npu

_mask_cache = {}
_MASK_CACHE_MAX_SIZE = 1


def _get_processed_mask(attention_mask, expand_mask):
    if attention_mask is None:
        return None
    cache_key = (attention_mask.data_ptr(), tuple(attention_mask.shape),
                 attention_mask.dtype, expand_mask)
    if cache_key in _mask_cache:
        return _mask_cache[cache_key]
    mask = attention_mask
    if expand_mask:
        mask = mask.expand(-1, -1, mask.size(-1), -1)
    mask = mask.to(torch.bool)
    if len(_mask_cache) >= _MASK_CACHE_MAX_SIZE:
        _mask_cache.clear()
    _mask_cache[cache_key] = mask
    return mask


def _make_output_forward_wrapper(old_forward):
    def new_forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        enable_boost = os.getenv("ENABLE_BOOST", "false").lower() in ["true", "1"]
        if enable_boost:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = torch_npu.npu_add_layer_norm(hidden_states,
                                                         input_tensor,
                                                         self.LayerNorm.weight,
                                                         self.LayerNorm.bias,
                                                         self.LayerNorm.eps,
                                                         )[0]
            return hidden_states
        else:
            return old_forward(self, hidden_states, input_tensor)

    return new_forward


custom_self_output_forward = _make_output_forward_wrapper
custom_output_forward = _make_output_forward_wrapper



def _npu_attention_forward_impl(self, hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        expand_mask: bool=True,
        *args,
        **kwargs) -> Tuple[torch.Tensor]:
    """
    NPU加速的注意力前向传播实现

    参数:
        self: 注意力层实例
        hidden_states: 输入张量
        attention_mask: 注意力掩码
        expand_mask: 是否扩展掩码维度

    返回:
        outputs: (输出张量,)
    """
    new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    query_layer = self.query(hidden_states).view(new_shape)
    key_layer = self.key(hidden_states).view(new_shape)
    value_layer = self.value(hidden_states).view(new_shape)

    if attention_mask is not None:
        attention_mask = _get_processed_mask(attention_mask, expand_mask)

    softmax_scale = 1 / math.sqrt(self.attention_head_size)
    out = torch_npu.npu_fused_infer_attention_score(
        query_layer,
        key_layer,
        value_layer,
        num_heads=self.num_attention_heads,
        input_layout="BSND",
        atten_mask=attention_mask,
        scale=softmax_scale)[0]
    out = out.view(hidden_states.size())

    return (out,)


def _make_attention_forward_wrapper(old_forward, expand_mask=True):
    """
    创建注意力前向传播的包装函数

    参数:
        old_forward: 原始的forward方法
        expand_mask: 是否扩展掩码维度

    返回:
        新的forward函数
    """
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        enable_boost = os.getenv("ENABLE_BOOST", "false").lower() in ["true", "1"]
        if enable_boost:
            return _npu_attention_forward_impl(self, hidden_states, attention_mask, expand_mask,
                                               *args, **kwargs)
        else:
            return old_forward(
                self, hidden_states, attention_mask, *args,**kwargs)
    return new_forward


def custom_self_attention_forward(old_forward):
    """XLM-RoBERTa等模型的注意力前向传播（需要扩展掩码）"""
    return _make_attention_forward_wrapper(old_forward, expand_mask=True)


def custom_bert_sdpa_self_attention_forward(old_forward):
    """BERT SDPA注意力前向传播（不需要扩展掩码）"""
    return _make_attention_forward_wrapper(old_forward, expand_mask=False)
