# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import base64
import json
from pathlib import Path

from loguru import logger

from mx_rag.utils.url import RequestUtils
from mx_rag.utils.file_check import FileCheck, SecFileCheck


class Img2ImgMultiModel:
    MAX_IMAGE_SIZE = 100 * 1024 * 1024
    SUPPORT_IMG_TYPE = (".jpg", ".png")
    HEADER = {
        'Content-Type': 'application/json'
    }
    IMAGE_ITEM = "image"

    def __init__(self, url: str, model_name=None, timeout: int = 10, max_prompt_len=1000, use_http: bool = False):
        self._url = url
        self._model_name = model_name
        self._client = RequestUtils(timeout=timeout, use_http=use_http)
        self._max_prompt_len = max_prompt_len

    def img2img(self, prompt: str, img_path: str) -> dict:
        resp = {"prompt": prompt, "result": ""}

        if prompt is None:
            logger.error(f"prompt cannot be None")
            return resp

        if len(prompt) > self._max_prompt_len or len(prompt) == 0:
            logger.error(f"prompt content len [{len(prompt)}] not in (0, {self._max_prompt_len}]")
            return resp

        FileCheck.check_path_is_exist_and_valid(img_path)
        SecFileCheck(img_path, self.MAX_IMAGE_SIZE).check()
        if Path(img_path).suffix not in self.SUPPORT_IMG_TYPE:
            raise TypeError(f"check [{img_path}] failed because the file type not be supported")

        with open(img_path, "rb") as f:
            content = f.read()
            encode_content = base64.b64encode(content)
            payload = {
                "prompt": prompt,
                self.IMAGE_ITEM: encode_content.decode()
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
