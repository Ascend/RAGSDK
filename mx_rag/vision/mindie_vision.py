# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import io
import json
from urllib.parse import urljoin
from loguru import logger

from mx_rag.utils import RequestUtils

HEADER = {
    'Content-Type': 'application/json'
}


class MindieVision:
    def __init__(self, model_name, url, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.url = url

    def text2img(self, prompt: str, output_format: str = "png"):
        request_body = {"prompt": prompt, "output_format": output_format}
        img_url = urljoin(self.url, 'text2img')
        request_util = RequestUtils()
        response = request_util.post(url=img_url, body=json.dumps(request_body), headers=HEADER)
        img_data = None
        if response.success:
            try:
                img_data = io.BytesIO(response.data)
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
        else:
            logger.error("get response failed")
        return img_data
