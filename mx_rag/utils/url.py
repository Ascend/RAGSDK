# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from ssl import SSLContext
from typing import Dict, Iterator

import urllib3
from loguru import logger

from mx_rag.libs.glib.checker.url_checker import HttpUrlChecker, HttpsUrlChecker
from .cert import TlsConfig
from .cert_check import CertContentsChecker
from .common import UrlUtilException
from .file_check import FileCheck, SecFileCheck

LIMIT_1M_SIZE = 1024 * 1024

HTTP_SUCCESS = 200


class UrlError(Exception):
    pass


class Result:
    def __init__(self, success: bool, data):
        self.success = success
        self.data = data


def _is_url_valid(url, use_http) -> bool:
    if url.startswith("http:") and not use_http:
        raise UrlError("http protocol is not support")
    check_key = "url"
    if use_http and HttpUrlChecker(check_key).check({check_key: url}):
        return True
    elif not use_http and HttpsUrlChecker(check_key).check({check_key: url}):
        return True
    return False


class RequestUtils:
    MAX_FILE_SIZE = 1 * 1024 * 1024

    def __init__(self,
                 retries=3,
                 timeout=10,
                 num_pools=200,
                 maxsize=200,
                 response_limit_size=LIMIT_1M_SIZE,
                 cert_file: str = "",
                 crl_file: str = "",
                 use_http: bool = False,
                 proxy_url: str = "",
                 ssl_context: SSLContext = None):
        self.use_http = use_http
        if cert_file:
            FileCheck.check_path_is_exist_and_valid(cert_file)
            SecFileCheck(cert_file, self.MAX_FILE_SIZE).check()
            try:
                with open(cert_file, "r") as f:
                    ca_data = f.read()
            except Exception as e:
                logger.warning(f"read cert file failed, find exception: {e}")
                raise UrlUtilException('read cert file failed') from e

            ret = CertContentsChecker("cert").check_dict({"cert": ca_data})
            if not ret:
                logger.error(f"invalid mef ca cert content: {ret.reason}")
                raise UrlUtilException('invalid cert content')

            if crl_file:
                FileCheck.check_path_is_exist_and_valid(crl_file)
                SecFileCheck(crl_file, self.MAX_FILE_SIZE).check()

            success, ssl_ctx = TlsConfig.get_client_ssl_context(cert_file, crl_file)
            if not success:
                raise UrlUtilException('unable to add ca_file for request')
        elif ssl_context:
            ssl_ctx = ssl_context
        else:
            ssl_ctx = TlsConfig.get_init_context()

        if proxy_url:
            self.pool = urllib3.ProxyManager(proxy_url=proxy_url,
                                             ssl_context=ssl_ctx,
                                             retries=retries,
                                             timeout=timeout,
                                             num_pools=num_pools,
                                             maxsize=maxsize)
        else:
            self.pool = urllib3.PoolManager(ssl_context=ssl_ctx,
                                            retries=retries,
                                            timeout=timeout,
                                            num_pools=num_pools,
                                            maxsize=maxsize)
        self.response_limit_size = response_limit_size

    def post(self, url: str, body: str, headers: Dict):
        if not _is_url_valid(url, self.use_http):
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
        if not _is_url_valid(url, self.use_http):
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

    def get(self, url: str, headers: Dict):
        if not _is_url_valid(url, self.use_http):
            logger.error(f"url check failed, url: {url}, use_http: {self.use_http}")
            return ""

        try:
            response = self.pool.request(method='GET',
                                         url=url,
                                         headers=headers,
                                         preload_content=False)
        except Exception as e:
            logger.error(f"request {url} failed, find exception: {e}")
            return ""
        if response.headers.get('Content-Type').find("text/html") == -1:
            logger.warning(f"The Content-Type in the response headers is not text/html, skip url: {url}.")
            return ""
        if response.status == HTTP_SUCCESS:
            try:
                return response.data
            except Exception as e:
                logger.error(f"read response failed, find exception: {e}")
                return ""
        else:
            logger.error(f"request failed with status code {response.status}")
            return ""

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
