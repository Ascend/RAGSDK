# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import io
import json
from urllib.parse import urljoin
from loguru import logger

from mx_rag.utils import RequestUtils


class MindieSD:
    HEADER = {
        'Content-Type': 'application/json'
    }
    MAX_PROMPT_LEN = 10000

    def __init__(self, model_name, url, timeout: int = 10):
        self.model_name = model_name
        self.url = url
        self.client = RequestUtils(timeout=timeout)

    def text2img(self, prompt: str, output_format: str = "png"):
        img_data = None
        if prompt is None or len(prompt) > MindieSD.MAX_PROMPT_LEN:
            logger.error(f"query cannot be None or content len not in (0, {MindieSD.MAX_PROMPT_LEN}]")
            return img_data
        request_body = {"prompt": prompt, "output_format": output_format}
        img_url = urljoin(self.url, 'text2img')
        response = self.client.post(url=img_url, body=json.dumps(request_body), headers=MindieSD.HEADER)
        if response.success:
            try:
                img_data = io.BytesIO(response.data)
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
        else:
            logger.error("get response failed")
        return img_data
