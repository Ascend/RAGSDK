# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
from urllib.parse import urljoin

from loguru import logger
import numpy as np

from mx_rag.utils.url import RequestUtils


class TEIReranker:
    HEADERS = {
        'Content-Type': 'application/json'
    }
    TEXT_MAX_LEN = 1000

    def __init__(self, url: str):
        self.url = urljoin(url, 'rerank')
        self.client = RequestUtils()

    @staticmethod
    def _process_data(scores_json, scores_len):
        if len(scores_json) != scores_len:
            raise Exception('tei response has different data length with request')

        scores = [0] * scores_len
        for score_json in scores_json:
            scores[score_json['index']] = score_json['score']

        return scores

    def rerank(self,
               query: [str],
               texts: list[str],
               batch_size: int = 32):
        texts_len = len(texts)
        if texts_len == 0:
            return np.array([])
        elif texts_len > TEIReranker.TEXT_MAX_LEN:
            logger.error(f'texts list length must less than {TEIReranker.TEXT_MAX_LEN}')
            return np.array([])

        result = []
        for start_index in range(0, texts_len, batch_size):
            texts_batch = texts[start_index: start_index + batch_size]

            request_body = {'query': query, 'texts': texts_batch, 'truncate': True}

            resp = self.client.post(self.url, json.dumps(request_body), headers=TEIReranker.HEADERS)

            if resp.success:
                try:
                    scores_json = json.loads(resp.data)
                    scores = self._process_data(scores_json, len(texts_batch))
                    result = result + scores
                except Exception as e:
                    logger.error(f'unable to process tei response content, find exception {e}')
                    return np.array([])
            else:
                logger.error('tei request failed')
                return np.array([])

        return np.array(result)
