# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import re
import json
from typing import List

from langchain_core.embeddings import Embeddings
from loguru import logger

from mx_rag.utils import ClientParam
from mx_rag.utils.common import (
    MAX_URL_LENGTH,
    EMBBEDDING_IMG_COUNT,
    EMBBEDDING_TEXT_COUNT,
    STR_MAX_LEN,
    MAX_BATCH_SIZE,
    MB
)
from mx_rag.utils.common import validate_params, validata_list_str, check_api_key
from mx_rag.utils.file_check import FileCheckError, PathNotFileException
from mx_rag.utils.url import RequestUtils


def _validate_image_data_uri(image_data_uri):
    # 正则表达式模式
    pattern = r'^data:image\/(jpeg|png|gif|bmp|tiff|webp);base64,[A-Za-z0-9+/=]+$'
    # 校验字符串
    match = re.match(pattern, image_data_uri)
    # 返回校验结果
    return match is not None


class CLIPEmbeddingError(Exception):
    pass


class CLIPEmbedding(Embeddings):
    HEADERS = {'Content-Type': 'application/json'}

    @validate_params(
        url=dict(validator=lambda x: isinstance(x, str) and 0 <= len(x) <= MAX_URL_LENGTH,
                 message=f"param must be str and str length range [0, {MAX_URL_LENGTH}]"),
        api_key=dict(validator=lambda x: check_api_key(x),
                     message="api_key check failed, please see the log"),
        client_param=dict(validator=lambda x: isinstance(x, ClientParam),
                          message="param must be instance of ClientParam"))
    def __init__(self, url: str, api_key: str = "", client_param=ClientParam()):
        self.url = url
        self.client = None
        try:
            self.client = RequestUtils(client_param=client_param)
        except FileCheckError as e:
            logger.error(f"CLIP client file param check failed:{e}")
        except PathNotFileException as e:
            logger.error(f"CLI client crt is not a file:{e}")
        except Exception as e:
            raise CLIPEmbeddingError('CLIP client init failed') from e

        if api_key != "" and not client_param.use_http:
            self.HEADERS['Authorization'] = "Bearer {}".format(api_key)

    @staticmethod
    def create(**kwargs):
        if "url" not in kwargs or not isinstance(kwargs.get("url"), str):
            logger.error("url param error. ")
            return None
        return CLIPEmbedding(**kwargs)

    @validate_params(
        texts=dict(
            validator=lambda x: validata_list_str(x, [1, EMBBEDDING_TEXT_COUNT], [1, STR_MAX_LEN]),
            message=f"param must meets: Type is List[str], list length range [1, {EMBBEDDING_TEXT_COUNT}], "
                    f"str length range [1, {STR_MAX_LEN}]"),
        batch_size=dict(
            validator=lambda x: 1 <= x <= MAX_BATCH_SIZE, message=f"param value range [1, {MAX_BATCH_SIZE}]"))
    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        return self._encode(texts, batch_size=batch_size)

    @validate_params(
        text=dict(
            validator=lambda x: 1 <= len(x) <= STR_MAX_LEN,
            message=f"param length range [1, {STR_MAX_LEN}]"))
    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_documents([text])
        if not embeddings:
            raise ValueError("Failed to embed text")
        return embeddings[0]

    @validate_params(
        images=dict(
            validator=lambda x: validata_list_str(x, [1, EMBBEDDING_IMG_COUNT], [1, 10 * MB]),
            message=f"param must meets: Type is List[str], list length range [1, {EMBBEDDING_IMG_COUNT}],"
                    f" str length range [1, {10 * MB}]"),
        batch_size=dict(
            validator=lambda x: 1 <= x <= MAX_BATCH_SIZE, message=f"param value range [1, {MAX_BATCH_SIZE}]"))
    def embed_images(self, images: List[str], batch_size: int = 32) -> List[List[float]]:
        if not all(_validate_image_data_uri(image_uri) for image_uri in images):
            raise ValueError("wrong image string, it must match "
                             "r'^data:image\/(jpeg|png|gif|bmp|tiff|webp);base64,[A-Za-z0-9+/=]+$'")
        return self._encode(uris=images, batch_size=batch_size)

    def _encode(self, texts: List[str] = None, uris: List[str] = None, batch_size: int = 32) -> List[List[float]]:
        texts = texts or []
        uris = uris or []

        inputs = [{'text': text} for text in texts]
        inputs.extend([{'uri': uri} for uri in uris])
        result = []

        for start_index in range(0, len(inputs), batch_size):
            batched_inputs = inputs[start_index: start_index + batch_size]
            request_body = {"data": batched_inputs}
            try:
                resp = self.client.post(self.url, json.dumps(request_body), headers=self.HEADERS)
                if not resp.success:
                    raise CLIPEmbeddingError('failed to get response')
                resp_data = json.loads(resp.data or "{}")
                if not isinstance(resp_data, dict) or "data" not in resp_data:
                    raise TypeError("Response data should be a dictionary containing a 'data' key")

                if len(resp_data["data"]) != len(batched_inputs):
                    raise ValueError("Response data length does not match input batch size")

                embeddings = [item['embedding'] for item in resp_data["data"] if 'embedding' in item]
                result.extend(embeddings)

            except json.JSONDecodeError as e:
                logger.error(f"failed to parse json response for batch starting at {start_index}: {e}")
                raise CLIPEmbeddingError(f"unable to parse clip response content: {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error while processing batch starting at {start_index}: {e}")
                raise CLIPEmbeddingError(
                    f"Failed to process CLIP response for batch starting at {start_index}: {e}") from e

        return result
