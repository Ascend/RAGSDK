# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import Optional

from loguru import logger
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, is_torch_npu_available

import mx_rag.utils as m_utils


class LocalEmbedding:
    TEXT_MAX_LEN = 1000

    def __init__(self,
                 model_name_or_path: str,
                 use_fp16: bool = True,
                 pooling_method: Optional[str] = None):
        self.model_name_or_path = model_name_or_path
        if self.model_name_or_path.startswith('/'):
            m_utils.dir_check(self.model_name_or_path)
        else:
            raise Exception('model_name_or_path must be an absolute path')

        self.pooling_method = 'cls'
        if pooling_method is not None:
            self.pooling_method = pooling_method

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, local_files_only=True)

        if use_fp16:
            self.model = self.model.half()

        try:
            if is_torch_npu_available():
                self.model.to('npu')
        except ImportError:
            logger.warning('unable to import torch_npu, please check if torch_npu is properly installed. '
                           'currently running on cpu.')

        self.model = self.model.eval()

    def encode(self,
               texts: list[str],
               batch_size: int = 32,
               max_length: int = 512):
        if len(texts) == 0:
            return np.array([])
        elif len(texts) > LocalEmbedding.TEXT_MAX_LEN:
            logger.error(f'texts list length must less than {LocalEmbedding.TEXT_MAX_LEN}')
            return np.array([])

        result = []
        for start_index in range(0, len(texts), batch_size):
            batch_texts = texts[start_index:start_index + batch_size]

            encode_texts = self.tokenizer(
                batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(
                self.model.device)

            with torch.no_grad():
                model_output = self.model(
                    encode_texts.input_ids, encode_texts.attention_mask, return_dict=True)
            last_hidden_state = model_output.last_hidden_state

            embeddings = self.pooling(last_hidden_state, encode_texts.attention_mask)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu().numpy().tolist()
            result = result + embeddings

        return np.array(result)

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        else:
            raise NotImplementedError(f'Pooling method {self.pooling_method} not implemented!')
