# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import base64
import json
from urllib.parse import urljoin

from loguru import logger

from mx_rag.utils import RequestUtils


class Text2ImgMultiModel:
    HEADER = {
        'Content-Type': 'application/json'
    }

    def __init__(self, url: str, model_name: str = None, timeout: int = 10, max_prompt_len=1000):
        self._model_name = model_name
        self._url = url
        self._client = RequestUtils(timeout=timeout)
        self._max_prompt_len = max_prompt_len

    def text2img(self, prompt: str, output_format: str = "png"):
        resp = {"prompt": prompt, "result": ""}
        if prompt is None:
            logger.error(f"prompt cannot be None")
            return resp

        if len(prompt) > self._max_prompt_len or len(prompt) == 0:
            logger.error(f"prompt content len [{len(prompt)}] not in (0, {self._max_prompt_len}]")
            return resp

        if output_format.lower() not in ["png", "jpeg", "jpg", "webp"]:
            logger.error("output format are not valid")
            return resp

        request_body = {"prompt": prompt, "output_format": output_format}
        img_url = urljoin(self._url, 'text2img')
        response = self._client.post(url=img_url, body=json.dumps(request_body), headers=self.HEADER)
        if not response.success:
            logger.error("text to generate image failed")
            return resp

        resp["result"] = base64.b64encode(response.data)
        return resp
