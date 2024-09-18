# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import base64
import json
import re

from loguru import logger

from mx_rag.utils.common import validate_params, INT_32_MAX, MAX_PROMPT_LENGTH, MAX_URL_LENGTH, \
    MAX_MODEL_NAME_LENGTH, MB, BOOL_TYPE_CHECK_TIP
from mx_rag.utils.url import RequestUtils


class Text2ImgMultiModel:
    HEADER = {
        'Content-Type': 'application/json'
    }

    @validate_params(
        url=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= MAX_URL_LENGTH,
                 message="param must be str and length range (0, 128]"),
        model_name=dict(validator=lambda x: x is None or isinstance(x, str) and 0 < len(x) <= MAX_MODEL_NAME_LENGTH,
                        message="param must be None or str, and str length range (0, 128]"),
        timeout=dict(validator=lambda x: isinstance(x, int) and 0 < x <= INT_32_MAX,
                     message="param must be str and length range (0, 128]"),
        use_http=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP),
        response_limit_size=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 10 * MB,
                                 message="param must be int and value range (0, 10 * MB]"),
    )
    def __init__(self, url: str, model_name: str = None, timeout: int = 10, use_http: bool = False,
                 response_limit_size: int = 1 * MB):
        self._model_name = model_name
        self._url = url
        self._client = RequestUtils(timeout=timeout, use_http=use_http, response_limit_size=response_limit_size)

    @validate_params(
        prompt=dict(validator=lambda x: 0 < len(x) <= MAX_PROMPT_LENGTH,
                    message="param must be str and length range (0, 1 * 1024 * 1024]"),
        output_format=dict(validator=lambda x: x in ["png", "jpeg", "jpg", "webp"],
                           message="param must be one of 'png', 'jpeg', 'jpg', 'webp'"),
        size=dict(validator=lambda x: re.compile(r"^\d{1,5}\*\d{1,5}$").match(x) is not None,
                  message=r"param must match '^\d{1,5}\*\d{1,5}$'"),
    )
    def text2img(self, prompt: str, output_format: str = "png", size: str = "512*512"):
        resp = {"prompt": prompt, "result": ""}

        request_body = {
            "prompt": prompt,
            "output_format": output_format,
            "size": size,
            "model_name": self._model_name
        }
        response = self._client.post(url=self._url, body=json.dumps(request_body), headers=self.HEADER)
        if not response.success:
            logger.error("text to generate image failed")
            return resp

        resp["result"] = response.data
        return resp
