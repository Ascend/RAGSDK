# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
from typing import List
from urllib.parse import urljoin

from loguru import logger
import numpy as np

from mx_rag.reranker.reranker import Reranker
from mx_rag.utils.url import RequestUtils


class TEIReranker(Reranker):
    HEADERS = {
        'Content-Type': 'application/json'
    }
    TEXT_MAX_LEN = 1000 * 1000

    def __init__(self, url: str, use_http: bool = False, k: int = 1):
        super(TEIReranker, self).__init__(k)
        self.url = urljoin(url, 'rerank')
        self.client = RequestUtils(use_http=use_http)

    @staticmethod
    def _process_data(scores_json, scores_len):
        if len(scores_json) != scores_len:
            raise ValueError('tei response has different data length with request')

        scores = [0.0] * scores_len
        visited = [False] * scores_len
        for score_json in scores_json:
            idx = score_json['index']
            sco = score_json['score']
            if not isinstance(idx, int):
                raise TypeError('index in tei response is not int value')
            if not isinstance(sco, float):
                raise TypeError('score in tei response it not float value')
            if idx >= scores_len or idx < 0:
                raise IndexError('index in tei response is not within valid range')
            if visited[idx]:
                raise ValueError('index in tei response is repeated')

            visited[idx] = True
            scores[idx] = sco
        return scores

    def rerank(self,
               query: [str],
               texts: List[str],
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
                    result.extend(scores)
                except Exception as e:
                    logger.error(f'unable to process tei response content, find exception {e}')
                    return np.array([])
            else:
                logger.error('tei request failed')
                return np.array([])

        return np.array(result)
