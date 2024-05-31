# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from loguru import logger
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, is_torch_npu_available

import mx_rag.utils as m_utils


class LocalReranker:
    TEXT_MAX_LEN = 1000

    def __init__(self,
                 model_name_or_path: str,
                 use_fp16: bool = True):
        self.model_name_or_path = model_name_or_path
        if self.model_name_or_path.startswith('/'):
            m_utils.dir_check(self.model_name_or_path)
        else:
            raise Exception('model_name_or_path must be an absolute path')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, local_files_only=True)

        if use_fp16:
            self.model = self.model.half()

        try:
            if is_torch_npu_available():
                self.model.to('npu')
        except ImportError:
            logger.warning('unable to import torch_npu, please check if torch_npu is properly installed. '
                           'currently running on cpu.')

        self.model = self.model.eval()

    def rerank(self,
               query: str,
               texts: list[str],
               batch_size: int = 32,
               max_length: int = 512):
        if len(texts) == 0:
            return np.array([])
        elif len(texts) > LocalReranker.TEXT_MAX_LEN:
            logger.error(f'texts list length must less than {LocalReranker.TEXT_MAX_LEN}')
            return np.array([])

        sentence_pairs = [[query, text] for text in texts]

        result = []
        for start_index in range(0, len(sentence_pairs), batch_size):
            sentence_batch = sentence_pairs[start_index:start_index + batch_size]

            encode_pairs = self.tokenizer(
                sentence_batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(
                self.model.device)

            with torch.no_grad():
                model_output = self.model(**encode_pairs, return_dict=True).logits.view(-1, ).float()

            scores = model_output.cpu().numpy().tolist()
            result = result + scores

        return np.array(result)
