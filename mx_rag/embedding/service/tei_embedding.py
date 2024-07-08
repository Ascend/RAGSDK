# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
from urllib.parse import urljoin

from loguru import logger
import numpy as np

from mx_rag.utils import RequestUtils
from mx_rag.embedding.embedding import Embedding


class TEIEmbedding(Embedding):
    HEADERS = {
        'Content-Type': 'application/json'
    }
    TEXT_MAX_LEN = 1000 * 1000

    def __init__(self, url: str, use_http: bool = False):
        self.url = urljoin(url, 'embed')
        self.client = RequestUtils(use_http=use_http)

    def embed_texts(self,
                    texts: list[str],
                    batch_size: int = 32):
        texts_len = len(texts)
        if texts_len == 0:
            return np.array([])
        elif texts_len > TEIEmbedding.TEXT_MAX_LEN:
            logger.error(f'texts list length must less than {TEIEmbedding.TEXT_MAX_LEN}')
            return np.array([])

        result = []
        for start_index in range(0, texts_len, batch_size):
            texts_batch = texts[start_index: start_index + batch_size]

            request_body = {'inputs': texts_batch, 'truncate': True}

            resp = self.client.post(self.url, json.dumps(request_body), headers=TEIEmbedding.HEADERS)

            if not resp.success:
                logger.error('tei request failed')
                return np.array([])

            try:
                data = json.loads(resp.data)
                if not isinstance(data, list):
                    raise TypeError('tei response is not list')
                if len(data) != len(texts_batch):
                    raise ValueError('tei response return data with different size')

                result.extend(data)
            except Exception as e:
                logger.error(f'unable to process tei response content, find exception {e}')
                return np.array([])

        return np.array(result)
