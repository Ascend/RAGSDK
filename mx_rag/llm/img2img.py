# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import json
import re

from loguru import logger

from mx_rag.utils.common import validate_params, INT_32_MAX, MAX_URL_LENGTH, \
    MAX_MODEL_NAME_LENGTH, MB, MAX_PROMPT_LENGTH, BOOL_TYPE_CHECK_TIP
from mx_rag.utils.url import RequestUtils


class Img2ImgMultiModel:
    HEADER = {
        'Content-Type': 'application/json'
    }
    IMAGE_ITEM = "image"

    @validate_params(
        url=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_URL_LENGTH,
                 message="param must be str and length range (0, 128]"),
        model_name=dict(validator=lambda x: x is None or isinstance(x, str) and 0 < len(x) <= MAX_MODEL_NAME_LENGTH,
                        message="param must be None or str, and str length range (0, 128]"),
        timeout=dict(validator=lambda x: isinstance(x, int) and 0 < x <= INT_32_MAX,
                     message="param must be int and value range (0, 2 ** 31 - 1]"),
        use_http=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP),
        response_limit_size=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 10 * MB,
                                 message="param must be int and value range (0, 10*MB]"),
    )
    def __init__(self, url: str, model_name=None, timeout: int = 10, use_http: bool = False,
                 response_limit_size=1 * MB):
        self._url = url
        self._model_name = model_name
        self._client = RequestUtils(timeout=timeout, use_http=use_http, response_limit_size=response_limit_size)

    @validate_params(
        prompt=dict(validator=lambda x: 0 < len(x) <= MAX_PROMPT_LENGTH,
                    message="param length range (0, 1 * 1024 * 1024]"),
        image_content=dict(validator=lambda x: 0 < len(x) <= 100 * MB,
                           message="param length range (0, 100 * 1024 * 1024]"),
        size=dict(validator=lambda x: re.compile(r"^\d{1,5}\*\d{1,5}$").match(x) is not None,
                  message=r"param must match '^\d{1,5}\*\d{1,5}$'"),
    )
    def img2img(self, prompt: str, image_content: str, size: str = "512*512") -> dict:
        resp = {"prompt": prompt, "result": ""}

        payload = {
            "prompt": prompt,
            self.IMAGE_ITEM: image_content,
            "size": size,
            "model_name": self._model_name
        }

        response = self._client.post(url=self._url, body=json.dumps(payload), headers=self.HEADER)
        if not response.success:
            logger.error("request img to generate img failed")
            return resp
        try:
            res = json.loads(response.data)
        except json.JSONDecodeError as e:
            logger.error(f"response content cannot convert to json format: {e}")
            return resp
        except Exception as e:
            logger.error(f"json load error: {e}")
            return resp

        if self.IMAGE_ITEM not in res:
            logger.error("request img to generate img failed, the response not contain image")
            return resp

        resp["result"] = res[self.IMAGE_ITEM]

        return resp
