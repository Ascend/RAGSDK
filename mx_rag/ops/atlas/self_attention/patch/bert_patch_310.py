# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import Optional, Tuple

import torch
import torch_npu
import mx_rag_opp
from transformers.models.bert.modeling_bert import BertSelfAttention


def check_transformer_version():
    import transformers
    expect_version = "4.41.1"
    version = transformers.__version__

    if version != expect_version:
        raise ImportError(f"transformers version must equal {expect_version} currents is {version}")


check_transformer_version()

old_self_attention_init = BertSelfAttention.__init__
old_self_attention_forward = BertSelfAttention.forward

enable_self_attention_speed: bool = True


def set_self_attention_enable_status(status :bool):
    global enable_self_attention_speed
    enable_self_attention_speed = status


def get_self_attention_enable_status():
    global enable_self_attention_speed
    return enable_self_attention_speed


class BertSelfAttentionSpeed:
    def __init__(self, config, position_embedding_type=None):
        self.speed_seq_len = [1024, 512, 256]

        self.query_w = None
        self.head_mask_defualt = None
        self.self_attention_op = mx_rag_opp.bert_self_attention()
        old_self_attention_init(self, config, position_embedding_type)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ) -> Tuple[torch.Tensor]:
        seq_len = hidden_states.size(1)
        if seq_len not in self.speed_seq_len or get_self_attention_enable_status() is False:
            return old_self_attention_forward(self, hidden_states, 
                attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, 
                past_key_value, output_attentions)

        if self.query_w is None:
            self.query_w = self.query.weight.data
            self.head_mask_defualt = torch.ones(1, self.num_attention_heads,
                 1, 1, dtype=self.query.weight.data.dtype).to(self.query_w.device)

        if head_mask is None:
            head_mask = self.head_mask_defualt

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        key_layer = self.key(hidden_states)
        new_x_shape = key_layer.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        key_layer = key_layer.view(new_x_shape)
        key_layer = key_layer.permute(0, 2, 3, 1)

        attention_scores = torch.matmul(query_layer, key_layer)

        attention_probs = self.self_attention_op.exec(
                attention_scores, head_mask, attention_mask, self.attention_head_size
            )
        
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


BertSelfAttention.__init__ = BertSelfAttentionSpeed.__init__
BertSelfAttention.forward = BertSelfAttentionSpeed.forward