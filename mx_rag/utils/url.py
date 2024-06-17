# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import Dict, Iterator

import urllib3
from loguru import logger

from mx_rag.libs.glib.checker.url_checker import HttpUrlChecker, HttpsUrlChecker
from .cert import TlsConfig
from .file_check import FileCheck

LIMIT_1M_SIZE = 1000 * 1000

HTTP_SUCCESS = 200


class Result:
    def __init__(self, success: bool, data):
        self.success = success
        self.data = data


def is_url_valid(url) -> bool:
    check_key = "url"
    if HttpUrlChecker(check_key).check({check_key: url}) or HttpsUrlChecker(check_key).check({check_key: url}):
        return True
    return False


class RequestUtils:

    def __init__(self,
                 retries=3,
                 timeout=10,
                 num_pools=200,
                 maxsize=200,
                 response_limit_size=LIMIT_1M_SIZE,
                 cert_file: str = ""):
        if cert_file:
            FileCheck.check_path_is_exist_and_valid(cert_file)
            success, ssl_ctx = TlsConfig.get_client_ssl_context(cert_file)
            if not success:
                raise Exception('unable to add ca_file for request')
        else:
            ssl_ctx = TlsConfig.get_init_context()

        self.pool = urllib3.PoolManager(ssl_context=ssl_ctx,
                                        retries=retries,
                                        timeout=timeout,
                                        num_pools=num_pools,
                                        maxsize=maxsize)
        self.response_limit_size = response_limit_size

    def post(self, url: str, body: str, headers: Dict):
        if not is_url_valid(url):
            logger.error("url check failed")
            return Result(False, "")

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

    def post_streamly(self, url: str, body: str, headers: Dict, chunk_size: int = 1024):
        if not is_url_valid(url):
            logger.error("url check failed")
            yield Result(False, "")

        try:
            response = self.pool.request(method='POST', url=url, body=body, headers=headers, preload_content=False)
        except Exception as e:
            logger.error(f"request {url} failed, find exception: {e}")
            yield Result(False, "")
            return

        try:
            content_type = str(response.headers.get("Content-Type"))
        except Exception as e:
            logger.error(f"get content type failed, find exception: {e}")
            yield Result(False, "")
            return

        if 'text/event-stream' not in content_type:
            logger.error("content type is not stream")
            yield Result(False, "")
            return

        if response.status == HTTP_SUCCESS:
            for result in self._iter_lines(response, chunk_size):
                yield result
        else:
            logger.error(f"request failed with status code {response.status}")
            yield Result(False, "")

    def _iter_lines(self, response, chunk_size=1024) -> Iterator[Result]:
        buffer = b''
        total_length = 0
        try:
            for chunk in response.stream(chunk_size):
                total_length += len(chunk)
                if total_length > self.response_limit_size:
                    logger.error("content length exceed limit")
                    yield Result(False, "")
                    return

                buffer += chunk
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    yield Result(True, line + b'\n')

            if buffer:
                yield Result(True, buffer)
        except Exception as e:
            logger.error(f"read response failed, find exception: {e}")
            yield Result(False, "")
