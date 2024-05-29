# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
from urllib.parse import urljoin

from loguru import logger

from mx_rag.utils import RequestUtils


class TEIEmbedding:
    HEADERS = {
        'Content-Type': 'application/json'
    }
    TEXT_MAX_LEN = 1000

    def __init__(self, url: str):
        self.url = urljoin(url, 'embed')
        self.client = RequestUtils()

    def encode(self,
               texts: list[str],
               batch_size: int = 32):
        texts_len = len(texts)
        if texts_len == 0:
            return []
        elif texts_len > TEIEmbedding.TEXT_MAX_LEN:
            logger.error(f'texts len must less than {TEIEmbedding.TEXT_MAX_LEN}')
            return []

        result = []
        for start_index in range(0, texts_len, batch_size):
            texts_batch = texts[start_index: start_index + batch_size]

            request_body = {'inputs': texts_batch, 'truncate': True}

            resp = self.client.post(self.url, json.dumps(request_body), headers=TEIEmbedding.HEADERS)

            if resp.success:
                try:
                    data = json.loads(resp.data)
                    result = result + data
                except Exception as e:
                    logger.error(f'unable to process tei response content, find exception {e}')
            else:
                logger.error('tei request failed')
                return []

        return result
