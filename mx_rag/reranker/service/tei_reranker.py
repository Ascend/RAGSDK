# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
from typing import List

from loguru import logger
import numpy as np

from mx_rag.reranker.reranker import Reranker
from mx_rag.utils import ClientParam
from mx_rag.utils.common import validate_params, MAX_TOP_K, INT_32_MAX, MAX_QUERY_LENGTH, TEXT_MAX_LEN, \
    validata_list_str, STR_TYPE_CHECK_TIP, MAX_API_KEY_LEN, STR_MAX_LEN
from mx_rag.utils.file_check import FileCheckError, PathNotFileException
from mx_rag.utils.url import RequestUtils


class TEIReranker(Reranker):
    HEADERS = {
        'Content-Type': 'application/json'
    }

    @validate_params(
        url=dict(validator=lambda x: isinstance(x, str), message=STR_TYPE_CHECK_TIP),
        k=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= MAX_TOP_K,
               message="param must be int and value range [1, 10000]"),
        api_key=dict(validator=lambda x: isinstance(x, str) and 0 <= len(x) <= MAX_API_KEY_LEN,
                     message="param must be str and str length range [0, 128]"),
        client_param=dict(validator=lambda x: isinstance(x, ClientParam),
                          message="param must be instance of ClientParam")
    )
    def __init__(self, url: str, k: int = 1, api_key: str = "", client_param=ClientParam()):
        super(TEIReranker, self).__init__(k)
        self.url = url
        self.client = None
        try:
            self.client = RequestUtils(client_param=client_param)
        except FileCheckError as e:
            logger.error(f"tei client file param check failed:{e}")
        except PathNotFileException as e:
            logger.error(f"tei client crt is not a file:{e}")
        except Exception:
            logger.error(f"init tei client failed")

        if api_key != "" and not client_param.use_http:
            self.HEADERS['Authorization'] = "Bearer {}".format(api_key)

    @staticmethod
    def create(**kwargs):
        if "url" not in kwargs or not isinstance(kwargs.get("url"), str):
            logger.error("url param error. ")
            return None

        return TEIReranker(**kwargs)

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

    @validate_params(
        query=dict(validator=lambda x: 1 <= len(x) <= MAX_QUERY_LENGTH,
                   message="param length range [1, 128 * 1024 * 1024]"),
        texts=dict(validator=lambda x: validata_list_str(x, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]),
                   message="param must meets: Type is List[str], "
                           "list length range [1, 1000 * 1000], str length range [1, 128 * 1024 * 1024]"),
        batch_size=dict(validator=lambda x: 1 <= x <= INT_32_MAX, message="param value range [1, 2 ** 31 - 1]")
    )
    def rerank(self,
               query: str,
               texts: List[str],
               batch_size: int = 32):
        texts_len = len(texts)
        result = []
        for start_index in range(0, texts_len, batch_size):
            texts_batch = texts[start_index: start_index + batch_size]

            request_body = {'query': query, 'texts': texts_batch, 'truncate': True}
            try:
                resp = self.client.post(self.url, json.dumps(request_body), headers=TEIReranker.HEADERS)
            except Exception as e:
                logger.error(f"API request failed with exception: {e}")
                return np.array([])
            if resp.success:
                try:
                    scores_json = json.loads(resp.data)
                    scores = self._process_data(scores_json, len(texts_batch))
                    result.extend(scores)
                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to decode JSON response from API: {json_err}")
                    return np.array([])
                except (TypeError, IndexError, ValueError) as e:
                    logger.error(f"Data processing error: {e}")
                    return np.array([])
                except Exception as e:
                    logger.error(f"Unable to process TEI response content, exception: {e}")
                    return np.array([])
            else:
                logger.error(f"TEI request failed.")
                return np.array([])

        return np.array(result)
