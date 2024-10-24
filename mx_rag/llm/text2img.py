# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import json
import re

from loguru import logger

from mx_rag.utils import ClientParam
from mx_rag.utils.common import validate_params, MAX_PROMPT_LENGTH, MAX_URL_LENGTH, MAX_MODEL_NAME_LENGTH
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
        client_param=dict(validator=lambda x: isinstance(x, ClientParam),
                          message="param must be instance of ClientParam"),
    )
    def __init__(self, url: str, model_name: str = None, client_param=ClientParam()):
        self._model_name = model_name
        self._url = url
        self._client = RequestUtils(client_param=client_param)

    @validate_params(
        prompt=dict(validator=lambda x: 0 < len(x) <= MAX_PROMPT_LENGTH,
                    message=f"param must be str and length range (0, {MAX_PROMPT_LENGTH}]"),
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
