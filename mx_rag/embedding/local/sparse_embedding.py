# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import List
from collections import defaultdict
import torch
import numpy as np

from langchain_core.embeddings import Embeddings
from loguru import logger
from transformers import AutoTokenizer, AutoModelForMaskedLM, is_torch_npu_available

from mx_rag.utils.common import validate_params, MAX_DEVICE_ID, TEXT_MAX_LEN, validata_list_str, \
    BOOL_TYPE_CHECK_TIP, STR_MAX_LEN, MAX_PATH_LENGTH, MAX_BATCH_SIZE, GB
from mx_rag.utils.file_check import SecDirCheck, safetensors_check

try:
    import torch_npu

    torch.npu.set_compile_mode(jit_compile=False)
except ImportError as e:
    logger.warning(f"Failed to import torch_npu: {e}. TextEmbedding will run on cpu.")
except Exception as e:
    logger.error(f"Unexpected error while importing torch_npu: {e}. TextEmbedding will run on cpu.")


class SparseEmbedding(Embeddings):
    @validate_params(
        model_path=dict(validator=lambda x: isinstance(x, str) and 0 <= len(x) <= MAX_PATH_LENGTH,
                        message="param must be str and str length range [0, 1024]"),
        dev_id=dict(validator=lambda x: isinstance(x, int) and 0 <= x <= MAX_DEVICE_ID,
                    message="param must be int and value range [0, 63]"),
        use_fp16=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP),
    )
    def __init__(self,
                 model_path: str,
                 dev_id: int = 0,
                 embedding_method: str = 'last',
                 use_fp16: bool = True):
        self.model_path = model_path
        SecDirCheck(self.model_path, 10 * GB).check()
        safetensors_check(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=True)
        self.embedding_method = embedding_method

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
            logger.error("model_path param error. ")
            return None

        return SparseEmbedding(**kwargs)

    @validate_params(
        texts=dict(validator=lambda x: validata_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                   message="param must meets: Type is List[str], list length range [1, 1000 * 1000], "
                           "str length range [1, 128 * 1024 * 1024]"),
        batch_size=dict(validator=lambda x: 1 <= x <= MAX_BATCH_SIZE,
                        message=f"param value range [1, {MAX_BATCH_SIZE}]"),
        max_length=dict(validator=lambda x: 1 <= x <= STR_MAX_LEN,
                        message=f"param value range [1, {STR_MAX_LEN}]")
    )
    def embed_documents(self,
                        texts: List[str],
                        batch_size: int = 32,
                        max_length: int = 512) -> List[dict]:
        result = self._encode(texts, batch_size, max_length)
        if len(result) == 0:
            raise ValueError("embedding text error")

        return result

    @validate_params(
        text=dict(validator=lambda x: 1 <= len(x) <= STR_MAX_LEN, message="param value range [1, 128 * 1024 * 1024]"),
        max_length=dict(validator=lambda x: 1 <= x <= STR_MAX_LEN,
                        message=f"param value range [1, {STR_MAX_LEN}]")
    )
    def embed_query(self, text: str, max_length: int = 512) -> dict:
        embeddings = self.embed_documents([text], max_length=max_length)
        if not embeddings:
            raise ValueError("embedding text failed")

        return embeddings[0]

    def _compute_sparse_logit(self, logits: torch.Tensor, tokens) -> torch.Tensor:
        """
        计算稀疏向量：对 logits 使用 ReLU 和 log(1+x) 变换，并根据 attention_mask 进行调整
        """
        return torch.max(
            torch.log(1 + torch.relu(logits)) * tokens.attention_mask.unsqueeze(-1),
            dim=1)[0]

    def _compute_sparse_last_hidden(self, hidden_state: torch.Tensor):
        """
                方式一：使用线性层进行稀疏向量化
        """
        sparse_linear = torch.nn.Linear(in_features=hidden_state.size(-1),
                                        out_features=1).to(self.model.device)
        token_weights = torch.relu(sparse_linear(hidden_state))

        return token_weights

    def _process_token_weights(self, token_weights: np.ndarray, input_ids: list):
        # conver to dict
        result = defaultdict(int)
        unused_tokens = set([self.tokenizer.cls_token_id, self.tokenizer.eos_token_id,
                             self.tokenizer.pad_token_id, self.tokenizer.unk_token_id])
        for w, idx in zip(token_weights, input_ids):
            if idx not in unused_tokens and w > 0:
                idx = str(idx)
                if w > result[idx]:
                    result[idx] = w
        return result

    def _map_token_to_scores(self, vec: torch.Tensor, input_ids: torch.Tensor) -> List[dict]:
        """
        将每个 token_id 映射到对应的分数
        """
        token_scores = vec.cpu().tolist()
        input_ids = input_ids.cpu().tolist()

        result = []
        for i, token_score in enumerate(token_scores):
            token_id_dict = {idx: token_score[idx] for idx in input_ids[i] if token_score[idx] > 0}
            result.append(token_id_dict)
        return result

    def _process_batch(self, batch_texts: List[str], max_length: int) -> List[dict]:
        tokens = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length,
                                return_tensors='pt').to(self.model.device)
        output = self.model(**tokens, output_hidden_states=True)

        # 计算稀疏向量
        if self.embedding_method == 'logits':
            vec = self._compute_sparse_logit(output.logits, tokens)
            return self._map_token_to_scores(vec, tokens['input_ids'])
        elif self.embedding_method == 'last':
            all_lexical_weights = []
            batch_data = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                return_tensors='pt',
                max_length=max_length,
            )
            vec = self._compute_sparse_last_hidden(output.hidden_states[-1])
            token_weights = vec.squeeze(-1)
            all_lexical_weights.extend(list(map(self._process_token_weights, token_weights.detach().cpu().numpy(),
                                    batch_data['input_ids'].cpu().numpy().tolist())))
            return all_lexical_weights

    def _encode(self, texts: List[str], batch_size: int = 32, max_length: int = 512) -> List[dict]:
        result = []
        for start_index in range(0, len(texts), batch_size):
            batch_texts = texts[start_index:start_index + batch_size]
            batch_result = self._process_batch(batch_texts, max_length)
            result.extend(batch_result)
        return result

