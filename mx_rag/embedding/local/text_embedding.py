# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import List
from typing import Optional

import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from loguru import logger
from transformers import AutoTokenizer, AutoModel, is_torch_npu_available

from mx_rag.utils.common import validate_params, MAX_DEVICE_ID, INT_32_MAX, TEXT_MAX_LEN, validata_list_str
from mx_rag.utils.file_check import FileCheck

try:
    import torch_npu

    torch.npu.set_compile_mode(jit_compile=False)
except Exception as e:
    logger.warning(f"import torch_npu failed:{e}, text_embedding will running on cpu")


class TextEmbedding(Embeddings):
    @validate_params(
        model_path=dict(validator=lambda x: isinstance(x, str)),
        dev_id=dict(validator=lambda x: isinstance(x, int) and 0 <= x <= MAX_DEVICE_ID),
        use_fp16=dict(validator=lambda x: isinstance(x, bool)),
        pooling_method=dict(validator=lambda x: x in ["cls", "mean"])
    )
    def __init__(self,
                 model_path: str,
                 dev_id: int = 0,
                 use_fp16: bool = True,
                 pooling_method: str = 'cls'):
        self.model_path = model_path
        FileCheck.dir_check(self.model_path)
        self.pooling_method = pooling_method
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)

        if use_fp16:
            self.model = self.model.half()

        try:
            if is_torch_npu_available():
                self.model.to(f'npu:{dev_id}')
        except ImportError:
            logger.warning('unable to import torch_npu, please check if torch_npu is properly installed. '
                           'currently running on cpu.')

        self.model = self.model.eval()

    @staticmethod
    def create(**kwargs):
        if "model_path" not in kwargs or not isinstance(kwargs.get("model_path"), str):
            raise KeyError("model_path param error. ")

        return TextEmbedding(**kwargs)

    @validate_params(
        batch_size=dict(validator=lambda x: 1 <= x <= INT_32_MAX),
        max_length=dict(validator=lambda x: 1 <= x <= INT_32_MAX)
    )
    def embed_documents(self,
                        texts: List[str],
                        batch_size: int = 32,
                        max_length: int = 512) -> List[List[float]]:
        result, _ = self._encode(texts, batch_size, max_length, False)
        if result.size == 0:
            raise ValueError("embedding text error")

        return result.tolist()

    @validate_params(
        text=dict(validator=lambda x: 1 <= len(x) <= INT_32_MAX),
        max_length=dict(validator=lambda x: 1 <= x <= INT_32_MAX)
    )
    def embed_query(self, text: str, max_length: int = 512) -> List[float]:
        embeddings = self.embed_documents([text], max_length=max_length)
        if not embeddings:
            raise ValueError("embedding text failed")

        return embeddings[0]

    @validate_params(
        batch_size=dict(validator=lambda x: 1 <= x <= INT_32_MAX),
        max_length=dict(validator=lambda x: 1 <= x <= INT_32_MAX)
    )
    def embed_documents_with_last_hidden_state(self,
                                               texts: List[str],
                                               batch_size: int = 32,
                                               max_length: int = 512):
        return self._encode(texts, batch_size, max_length, True)

    @validate_params(
        texts=dict(validator=lambda x: validata_list_str(x, [1, TEXT_MAX_LEN], [1, INT_32_MAX]))
    )
    def _encode(self,
                texts: List[str],
                batch_size: int = 32,
                max_length: int = 512,
                with_last_hidden_state: bool = False):
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
                raise Exception('process last_hidden_state failed') from le

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
