# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
from typing import List

from langchain_core.embeddings import Embeddings

from mx_rag.utils.common import validate_params, INT_32_MAX, EMBBEDDING_TEXT_COUNT, validata_list_str
from mx_rag.utils.url import RequestUtils


class TEIEmbedding(Embeddings):
    HEADERS = {
        'Content-Type': 'application/json'
    }

    @validate_params(
        url=dict(validator=lambda x: isinstance(x, str)),
        use_http=dict(validator=lambda x: isinstance(x, bool))
    )
    def __init__(self, url: str, use_http: bool = False):
        self.url = url
        self.client = RequestUtils(use_http=use_http)
        self.use_http = use_http

    @staticmethod
    def create(**kwargs):
        if "url" not in kwargs or not isinstance(kwargs.get("url"), str):
            raise KeyError("url param error. ")

        return TEIEmbedding(**kwargs)

    @validate_params(
        texts=dict(validator=lambda x: validata_list_str(x, [1, EMBBEDDING_TEXT_COUNT], [1, INT_32_MAX])),
        batch_size=dict(validator=lambda x: 1 <= x <= INT_32_MAX)
    )
    def embed_documents(self,
                        texts: List[str],
                        batch_size: int = 32) -> List[List[float]]:

        texts_len = len(texts)
        result = []
        for start_index in range(0, texts_len, batch_size):
            texts_batch = texts[start_index: start_index + batch_size]

            request_body = {'inputs': texts_batch, 'truncate': True}

            resp = self.client.post(self.url, json.dumps(request_body), headers=self.HEADERS)

            if not resp.success:
                raise Exception("tei get response failed")

            try:
                data = json.loads(resp.data)
                if not isinstance(data, list):
                    raise TypeError('tei response is not list')
                if len(data) != len(texts_batch):
                    raise ValueError('tei response return data with different size')

                result.extend(data)
            except Exception as e:
                raise Exception('unable to process tei response content') from e

        return result

    @validate_params(
        text=dict(validator=lambda x: 1 <= len(x) <= INT_32_MAX)
    )
    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_documents([text])
        if not embeddings:
            raise ValueError("embedding text failed")

        return embeddings[0]
