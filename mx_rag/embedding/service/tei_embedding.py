#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
from typing import List

from langchain_core.embeddings import Embeddings
from loguru import logger

from mx_rag.utils import ClientParam
from mx_rag.utils.common import validate_params, EMBEDDING_TEXT_COUNT, validate_list_str, \
    STR_MAX_LEN, MAX_URL_LENGTH, MAX_BATCH_SIZE
from mx_rag.utils.file_check import FileCheckError, PathNotFileException
from mx_rag.utils.url import RequestUtils


class TEIEmbeddingError(Exception):
    pass


class TEIEmbedding(Embeddings):

    @validate_params(
        url=dict(validator=lambda x: isinstance(x, str) and 0 <= len(x) <= MAX_URL_LENGTH,
                 message="param must be str and str length range [0, 128]"),
        client_param=dict(validator=lambda x: isinstance(x, ClientParam),
                          message="param must be instance of ClientParam"),
        embed_mode=dict(validator=lambda x: isinstance(x, str) and x in ('dense', 'sparse'),
                        message=f"param must be str and in ('dense', 'sparse')"),
    )
    def __init__(self, url: str, client_param=ClientParam(), embed_mode: str = 'dense'):
        self.url = url
        self.embed_mode = embed_mode
        self.client = None
        self.headers = {
            'Content-Type': 'application/json'
        }
        try:
            self.client = RequestUtils(client_param=client_param)
        except FileCheckError as e:
            logger.error(f"tei client file param check failed:{e}")
        except PathNotFileException as e:
            logger.error(f"tei client crt is not a file:{e}")
        except Exception:
            logger.error(f"init tei client failed")

    @staticmethod
    def create(**kwargs):
        if "url" not in kwargs or not isinstance(kwargs.get("url"), str):
            logger.error("url param error. ")
            return None

        return TEIEmbedding(**kwargs)

    @validate_params(
        texts=dict(validator=lambda x: validate_list_str(x, [1, EMBEDDING_TEXT_COUNT], [1, STR_MAX_LEN]),
                   message="param must meets: Type is List[str], list length range [1, 1000 * 1000], "
                           "str length range [1, 128 * 1024 * 1024]"),
        batch_size=dict(validator=lambda x: 1 <= x <= MAX_BATCH_SIZE,
                        message=f"param value range [1, {MAX_BATCH_SIZE}]")
    )
    def embed_documents(self,
                        texts: List[str],
                        batch_size: int = 32) -> List[List[float]]:

        texts_len = len(texts)
        result = []
        for start_index in range(0, texts_len, batch_size):
            texts_batch = texts[start_index: start_index + batch_size]

            request_body = {'inputs': texts_batch, 'truncate': True}

            resp = self.client.post(self.url, json.dumps(request_body), headers=self.headers)

            if not resp.success:
                raise TEIEmbeddingError("tei get response failed")

            try:
                data = json.loads(resp.data)
                if not isinstance(data, list):
                    raise TypeError('tei response is not list')
                if len(data) != len(texts_batch):
                    raise ValueError('tei response return data with different size')
                if self.embed_mode == 'sparse':
                    data = [{item['index']: item['value'] for item in sub_list} for sub_list in data]

                result.extend(data)
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse JSON response for batch starting at {start_index}: {json_err}")
                raise TEIEmbeddingError(f"Unable to parse TEI response content: {json_err}") from json_err

            except (TypeError, ValueError) as data_err:
                logger.error(f"Error in TEI response data for batch starting at {start_index}: {data_err}")
                raise TEIEmbeddingError(f"TEI response data error: {data_err}") from data_err

            except Exception as e:
                logger.error(f"Unexpected error while processing batch starting at {start_index}: {e}")
                raise TEIEmbeddingError(
                    f"Failed to process TEI response for batch starting at {start_index}: {e}") from e

        return result

    @validate_params(
        text=dict(validator=lambda x: 1 <= len(x) <= STR_MAX_LEN, message="param length range [1, 128 * 1024 * 1024]")
    )
    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_documents([text])
        if not embeddings:
            raise ValueError("embedding text failed")

        return embeddings[0]
