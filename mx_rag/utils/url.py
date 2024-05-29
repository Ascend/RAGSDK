# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import Dict

from loguru import logger
import urllib3

from .cert import TlsConfig

LIMIT_1M_SIZE = 1000 * 1000

HTTP_SUCCESS = 200


class Result:
    def __init__(self, success: bool, data: str):
        self.success = success
        self.data = data


class RequestUtils:

    def __init__(self,
                 retries=3,
                 timeout=10,
                 num_pools=200,
                 maxsize=200,
                 response_limit_size=LIMIT_1M_SIZE):
        ssl_ctx = TlsConfig.get_init_context()
        self.pool = urllib3.PoolManager(ssl_context=ssl_ctx,
                                        retries=retries, timeout=timeout, num_pools=num_pools, maxsize=maxsize)
        self.response_limit_size = response_limit_size

    def post(self, url: str, body: str, headers: Dict):
        try:
            response = self.pool.request(method='POST',
                                         url=url,
                                         body=body,
                                         headers=headers,
                                         preload_content=False)
        except Exception as e:
            logger.error(f"request {url} failed, find exception: {e}")
            return Result(False, "")

        try:
            content_length = int(response.headers.get("Content-Length"))
        except Exception as e:
            logger.error(f"get content length failed, find exception: {e}")
            return Result(False, "")

        if content_length > self.response_limit_size:
            logger.error("content length exceed limit")
            return Result(False, "")

        if response.status == HTTP_SUCCESS:
            try:
                response_data = response.read(amt=self.response_limit_size)
            except Exception as e:
                logger.error(f"read response failed, find exception: {e}")
                return Result(False, "")

            return Result(True, response_data)
        else:
            logger.error(f"request failed with status code {response.status}")
            return Result(False, "")
