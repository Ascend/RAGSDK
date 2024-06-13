# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import List
from typing import Optional

import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModel, is_torch_npu_available

import mx_rag.utils as m_utils
from mx_rag.embedding.embedding import Embedding

try:
    import torch_npu

    torch.npu.set_compile_mode(jit_compile=False)
except Exception as e:
    logger.warning(f"import torch_npu failed:{e}, text_embedding will running on cpu")


class TextEmbedding(Embedding):
    TEXT_MAX_LEN = 1000

    def __init__(self,
                 model_name_or_path: str,
                 dev_id: int = 0,
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
                self.model.to(f'npu:{dev_id}')
        except ImportError:
            logger.warning('unable to import torch_npu, please check if torch_npu is properly installed. '
                           'currently running on cpu.')

        self.model = self.model.eval()

    def embed_texts(self,
                    texts: List[str],
                    batch_size: int = 32,
                    max_length: int = 512):

        result, _ = self._encode(texts, batch_size, max_length, False)
        return result

    def embed_texts_with_last_hidden_state(self,
                                           texts: list[str],
                                           batch_size: int = 32,
                                           max_length: int = 512):
        return self._encode(texts, batch_size, max_length, True)

    def _encode(self,
                texts: list[str],
                batch_size: int = 32,
                max_length: int = 512,
                with_last_hidden_state: bool = False):
        if len(texts) == 0:
            return np.array([]), np.array([])
        elif len(texts) > self.TEXT_MAX_LEN:
            logger.error(f'texts list length must less than {self.TEXT_MAX_LEN}')
            return np.array([]), np.array([])

        result = []
        last_hidden_states = []
        for start_index in range(0, len(texts), batch_size):
            batch_texts = texts[start_index:start_index + batch_size]

            encode_texts = self.tokenizer(
                batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(
                self.model.device)

            attention_mask = encode_texts.attention_mask
            with torch.no_grad():
                model_output = self.model(encode_texts.input_ids, attention_mask, return_dict=True)
            last_hidden_state = model_output.last_hidden_state
            embeddings = self._pooling(last_hidden_state, attention_mask)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu().numpy()
            result.append(embeddings)

            if not with_last_hidden_state:
                continue

            try:
                attention_mask = attention_mask.cpu()
                last_hidden_state = last_hidden_state.cpu().numpy()

                left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
                if left_padding:
                    last_hidden_state = last_hidden_state[:, -1]
                else:
                    sequence_length = attention_mask.sum(dim=1) - 1
                    current_batch_size = len(batch_texts)
                    last_hidden_state = last_hidden_state[torch.arange(current_batch_size), sequence_length]

                last_hidden_states.append(last_hidden_state)
            except Exception as le:
                logger.error(f'process last_hidden_state failed, find exception {le}')
                return np.array([]), np.array([])

        last_hidden_states = np.concatenate(last_hidden_states, axis=0) if with_last_hidden_state else np.array([])
        return np.concatenate(result, axis=0), last_hidden_states

    def _pooling(self,
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
