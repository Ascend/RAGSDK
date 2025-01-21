# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import List

import torch
from langchain_core.embeddings import Embeddings
from loguru import logger
from transformers import AutoTokenizer, AutoModelForMaskedLM, is_torch_npu_available

from mx_rag.utils.common import validate_params, MAX_DEVICE_ID, TEXT_MAX_LEN, validata_list_str, \
    BOOL_TYPE_CHECK_TIP, STR_MAX_LEN, MAX_PATH_LENGTH, MAX_BATCH_SIZE, validate_lock, GB
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
        lock=dict(
            validator=lambda x: x is None or validate_lock(x),
            message="param must be one of None, multiprocessing.Lock(), threading.Lock()")
    )
    def __init__(self,
                 model_path: str,
                 dev_id: int = 0,
                 use_fp16: bool = True):
        self.model_path = model_path
        SecDirCheck(self.model_path, 10 * GB).check()
        safetensors_check(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=True)

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

    def _encode(self,
                texts: List[str],
                batch_size: int = 32,
                max_length: int = 512):
        result = []
        for start_index in range(0, len(texts), batch_size):
            batch_texts = texts[start_index:start_index + batch_size]
            tokens = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length,
                                    return_tensors='pt').to(self.model.device)
            output = self.model(**tokens)
            vec = torch.max(
                torch.log(
                    1 + torch.relu(output.logits)
                ) * tokens.attention_mask.unsqueeze(-1),
                dim=1)[0]
            # 获取 token scores，并映射到实际 token
            token_scores = vec.cpu().tolist()
            input_ids = tokens['input_ids'].cpu().tolist()

            # 创建 token 和分数的映射字典
            token_id_list = []
            for i, token_score in enumerate(token_scores):
                token_id_dict = {}
                for idx in input_ids[i]:
                    if token_score[idx] > 0:
                        token_id_dict[idx] = token_score[idx]
                token_id_list.append(token_id_dict)
            result.extend(token_id_list)
        return result


