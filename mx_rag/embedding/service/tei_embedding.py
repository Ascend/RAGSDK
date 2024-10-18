# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
from typing import List

from langchain_core.embeddings import Embeddings
from loguru import logger

from mx_rag.utils import ClientParam
from mx_rag.utils.common import validate_params, INT_32_MAX, EMBBEDDING_TEXT_COUNT, validata_list_str, \
    STR_TYPE_CHECK_TIP, MAX_API_KEY_LEN, STR_MAX_LEN, MAX_URL_LENGTH
from mx_rag.utils.file_check import FileCheckError, PathNotFileException
from mx_rag.utils.url import RequestUtils


class TEIEmbeddingError(Exception):
    pass


class TEIEmbedding(Embeddings):
    HEADERS = {
        'Content-Type': 'application/json'
    }

    @validate_params(
        url=dict(validator=lambda x: isinstance(x, str) and 0 <= len(x) <= MAX_URL_LENGTH,
                 message="param must be str and str length range [0, 128]"),
        api_key=dict(validator=lambda x: isinstance(x, str) and 0 <= len(x) <= MAX_API_KEY_LEN,
                     message="param must be str and str length range [0, 128]"),
        client_param=dict(validator=lambda x: isinstance(x, ClientParam),
                          message="param must be instance of ClientParam"),
    )
    def __init__(self, url: str, api_key: str = "", client_param=ClientParam()):
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

        return TEIEmbedding(**kwargs)

    @validate_params(
        texts=dict(validator=lambda x: validata_list_str(x, [1, EMBBEDDING_TEXT_COUNT], [1, STR_MAX_LEN]),
                   message="param must meets: Type is List[str], list length range [1, 1000 * 1000], "
                           "str length range [1, 128 * 1024 * 1024]"),
        batch_size=dict(validator=lambda x: 1 <= x <= INT_32_MAX, message="param value range [1, 2 ** 31 - 1]")
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
                raise TEIEmbeddingError("tei get response failed")

            try:
                data = json.loads(resp.data)
                if not isinstance(data, list):
                    raise TypeError('tei response is not list')
                if len(data) != len(texts_batch):
                    raise ValueError('tei response return data with different size')

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
