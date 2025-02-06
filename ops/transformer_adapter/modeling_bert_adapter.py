# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import json
import os
from typing import Optional, List, Union, Tuple

import torch
import torch_npu
from transformers.models.bert.modeling_bert import BertModel, BertConfig, BaseModelOutputWithPoolingAndCrossAttentions


def load_acl_transformer():
    rag_sdk_home_path = os.getenv("RAG_SDK_HOME", "")
    if not rag_sdk_home_path or not os.path.exists(rag_sdk_home_path):
        raise RuntimeError("env RAG_SDK_HOME not exist, source ~/.bashrc")
    lib_path = os.path.join(rag_sdk_home_path, "ops/lib/libatb_torch.so")
    torch.classes.load_library(lib_path)


def is_nd():
    soc_version = torch_npu._C._npu_get_soc_version()
    return soc_version in [104, 220, 221, 222, 223, 224]


load_acl_transformer()
MASK_INC_DIM1 = 1 if is_nd() else 16

enable_self_attention_speed: bool = True

old_init = BertModel.__init__
old_forward = BertModel.forward


def new__init__(self, config, add_pooling_layer=True):
    self.boost_flag = int(os.getenv("ENABLE_BOOST", 0))
    self.max_seq_len = config.max_position_embeddings
    if self.boost_flag:
        old_init(self, config, add_pooling_layer)
        self.init_ascend_operations_boost(config)
        self.layer_id_list = [torch.tensor([i], dtype=torch.int32).npu() for i in range(config.num_hidden_layers)]
    else:
        old_init(self, config, add_pooling_layer)


def init_ascend_operations_boost(self, config: BertConfig):
    self.head_size = config.hidden_size // config.num_attention_heads
    self.head_num = config.num_attention_heads
    if hasattr(config, 'world_size'):
        rank = torch.distributed.get_rank()
        rank_size = torch.distributed.get_world_size()
        self.acl_param = json.dumps({"headNum": self.head_num, "layerNormEps": config.layer_norm_eps,
                                     "dk": self.head_size, "layerNum": config.num_hidden_layers, "rank": rank,
                                     "rankSize": rank_size})
    else:
        self.acl_param = json.dumps({"headNum": self.head_num, "layerNormEps": config.layer_norm_eps,
                                     "dk": self.head_size, "layerNum": config.num_hidden_layers})
    self.max_position_embeddings = config.max_position_embeddings

    self.acl_fa_operation = torch.classes.ModelTorch.ModelTorch("bge_large_FlashAttentionModel")

    self.acl_fa_operation.set_param(self.acl_param)

    self.num_layers = config.num_hidden_layers
    self.hidden_size = config.hidden_size
    self.ascend_weight = []
    self.min_cache = torch.full(
        (self.max_position_embeddings, self.max_position_embeddings),
        torch.finfo(torch.half).min, dtype=torch.half).npu()


def init_ascend_weight_boost(self):
    weights: List = []
    weights = [self.state_dict()["embeddings.word_embeddings.weight"],
               self.state_dict()["embeddings.position_embeddings.weight"],
               self.state_dict()["embeddings.token_type_embeddings.weight"],
               self.state_dict()["embeddings.LayerNorm.weight"],
               self.state_dict()["embeddings.LayerNorm.bias"]
               ]
    for i in range(self.num_layers):
        weights_t = []
        weights_layer = self.encoder.layer[i].state_dict()
        weights_t.append(weights_layer["attention.self.query.weight"].t().contiguous())
        weights_t.append(weights_layer["attention.self.query.bias"])
        weights_t.append(weights_layer["attention.self.key.weight"].t().contiguous())
        weights_t.append(weights_layer["attention.self.key.bias"])
        weights_t.append(weights_layer["attention.self.value.weight"].t().contiguous())
        weights_t.append(weights_layer["attention.self.value.bias"])
        weights_t.append(weights_layer["attention.output.dense.weight"].t().contiguous())
        weights_t.append(weights_layer["attention.output.dense.bias"])
        weights_t.append(weights_layer["attention.output.LayerNorm.weight"])
        weights_t.append(weights_layer["attention.output.LayerNorm.bias"])
        weights_t.append(weights_layer["intermediate.dense.weight"].t().contiguous())
        weights_t.append(weights_layer["intermediate.dense.bias"])
        weights_t.append(weights_layer["output.dense.weight"].t().contiguous())
        weights_t.append(weights_layer["output.dense.bias"])
        weights_t.append(weights_layer["output.LayerNorm.weight"])
        weights_t.append(weights_layer["output.LayerNorm.bias"])
        weights.extend(weights_t)
    self.ascend_weight = weights
    self.acl_fa_operation.set_weight(weights)


def prepare_inputs_for_ascend_boost(self, input_ids, position_ids, token_type_ids, attention_mask=None,
                                    past_key_values=None):
    batch_size, seq_len = input_ids.shape
    position_ids = position_ids.npu()
    token_type_ids = token_type_ids.npu()
    attention_mask = attention_mask.float().half()
    mask = attention_mask.clone()
    mask[mask == 0] = -65504.0
    mask[mask == 1] = -0.0
    attention_mask_max = torch.zeros(batch_size, self.max_seq_len, self.max_seq_len, device="npu", dtype=torch.half)
    for i in range(batch_size):
        attention_mask_max[i, :seq_len, :seq_len] = mask[i]
    token_offset_tensor = torch.full((batch_size,), seq_len, device="npu", dtype=torch.int32)
    seq_len_tensor = torch.full((batch_size,), seq_len, device="npu", dtype=torch.int32)

    inputs = [input_ids,
              position_ids,
              token_type_ids,
              attention_mask_max,
              token_offset_tensor,
              seq_len_tensor,
              ] + self.layer_id_list
    return inputs


def execute_ascend_operator_boost(self, input_ids, position_ids, token_type_ids, attention_mask=None,
                                  past_key_values=None):
    batch_size, seq_len = input_ids.shape
    acl_inputs = self.prepare_inputs_for_ascend_boost(input_ids, position_ids, token_type_ids, attention_mask,
                                                      past_key_values)
    tmp_param = json.dumps(
        {"tokenOffset": [seq_len] * batch_size,
         "seqLen": [seq_len] * batch_size
         })
    acl_model_out = self.acl_fa_operation.execute(acl_inputs, tmp_param)
    acl_hidden_state = acl_model_out[0]
    return acl_hidden_state


def geneate_position_ids(input_ids: Optional[torch.Tensor] = None,
                         inputs_embeds: Optional[torch.Tensor] = None,
                         position_ids: Optional[torch.Tensor] = None,
                         past_key_values_length:int = 0,
                         seq_length:int = 0):
    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    return position_ids


def geneate_token_type_ids(device, input_shape, embeddings, token_type_ids: Optional[torch.Tensor] = None):
    if token_type_ids is not None:
        return token_type_ids

    if hasattr(embeddings, "token_type_ids"):
        batch_size, seq_length = input_shape
        buffered_token_type_ids = embeddings.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        token_type_ids = buffered_token_type_ids_expanded
    else:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    return token_type_ids


def forward_boost(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = None,
) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
    if self.boost_flag:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        past_key_values_length = 0
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        position_ids = geneate_position_ids(input_ids, inputs_embeds, position_ids, past_key_values_length, seq_length)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        token_type_ids = geneate_token_type_ids(device, input_shape, self.embeddings, token_type_ids)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # add acl model
        if not self.ascend_weight:
            self.init_ascend_weight_boost()

        hidden_states = self.execute_ascend_operator_boost(input_ids,
                                                           position_ids,
                                                           token_type_ids,
                                                           attention_mask,
                                                           past_key_values)

        sequence_output = hidden_states
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )
    else:
        return old_forward(self,
                           input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           head_mask=head_mask,
                           inputs_embeds=inputs_embeds,
                           past_key_values=past_key_values,
                           output_attentions=output_attentions,
                           output_hidden_states=output_hidden_states,
                           return_dict=return_dict)


BertModel.__init__ = new__init__
BertModel.init_ascend_operations_boost = init_ascend_operations_boost
BertModel.init_ascend_weight_boost = init_ascend_weight_boost
BertModel.prepare_inputs_for_ascend_boost = prepare_inputs_for_ascend_boost
BertModel.execute_ascend_operator_boost = execute_ascend_operator_boost
BertModel.forward = forward_boost
