# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import Dict, Iterator

import urllib3
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from urllib3.exceptions import TimeoutError as urllib3_TimeoutError, HTTPError
from loguru import logger

from glib.checker import HttpUrlChecker, HttpsUrlChecker
from mx_rag.utils.client_param import ClientParam
from .cert_check import CertContentsChecker
from .common import MB
from .file_check import SecFileCheck
from .tls_cfg import get_two_way_auth_ssl_context, get_default_context, get_one_way_auth_ssl_context

HTTP_SUCCESS = 200
MAX_CERT_FILE_SIZE = MB
MIN_PASSWORD_LENGTH = 8
PASSWORD_REQUIREMENT = 2


class Result:
    def __init__(self, success: bool, data):
        self.success = success
        self.data = data


def is_url_valid(url, use_http) -> bool:
    if url.startswith("http:") and not use_http:
        return False
    check_key = "url"
    if use_http and HttpUrlChecker(check_key).check({check_key: url}):
        return True
    elif not use_http and HttpsUrlChecker(check_key).check({check_key: url}):
        return True
    return False


class RequestUtils:

    def __init__(self,
                 retries=3,
                 num_pools=200,
                 maxsize=200,
                 client_param: ClientParam = ClientParam()
                 ):

        self.use_http = client_param.use_http
        self.response_limit_size = client_param.response_limit_size

        if not client_param.use_http:
            # 未配置ssl context并且使用https时，校验证书相关参数合法性
            self._check_https_para(client_param)
            # 配置双向认证
            if client_param.key_file or client_param.crt_file or client_param.pwd:
                ssl_ctx = get_two_way_auth_ssl_context(client_param)
            else:
                # 配置单向认证
                ssl_ctx = get_one_way_auth_ssl_context(client_param)
        else:
            ssl_ctx = get_default_context()

        self.pool = urllib3.PoolManager(ssl_context=ssl_ctx,
                                        retries=retries,
                                        timeout=client_param.timeout,
                                        num_pools=num_pools,
                                        maxsize=maxsize)

    @staticmethod
    def _check_ca_content(ca_file: str):
        try:
            with open(ca_file, "r") as f:
                ca_data = f.read()
        except FileNotFoundError as e:
            logger.error(f"Certificate file '{ca_file}' not found.")
            raise ValueError(f"Certificate file '{ca_file}' not found.") from e
        except PermissionError as e:
            logger.error(f"Permission denied when reading certificate file: '{ca_file}'")
            raise ValueError(f"Permission denied for certificate file: {ca_file}") from e
        except Exception as e:
            logger.error(f"read cert file failed, find exception: {e}")
            raise ValueError('read cert file failed') from e

        ret = CertContentsChecker("cert").check_dict({"cert": ca_data})
        if not ret:
            logger.error(f"invalid ca cert content: '{ret.reason}'")
            raise ValueError('invalid cert content')

    @staticmethod
    def _check_password(plain_text: str):
        if not plain_text:
            raise ValueError("Invalid password length.")

        # Initialize flags for character types
        has_lower = has_upper = has_digits = has_symbol = False

        # Iterate through each character in the plain text
        for char in plain_text:
            if char.islower():
                has_lower = True
            elif char.isupper():
                has_upper = True
            elif char.isdigit():
                has_digits = True
            else:
                has_symbol = True

        # Check if password meets requirements
        if len(plain_text) < MIN_PASSWORD_LENGTH and \
                (has_lower + has_upper + has_digits + has_symbol) < PASSWORD_REQUIREMENT:
            logger.warning("The password is too weak. It should contain at least two of the following:"
                           " lowercase characters, uppercase characters, numbers, and symbols,"
                           " and the password must contain at least %d characters. ", MIN_PASSWORD_LENGTH)

    @staticmethod
    def _check_key_file_whether_encrypted(key_path: str):
        def check(file: str):
            try:
                # 无密码方式加载key文件
                with open(key_path, 'rb') as fi:
                    key = load_pem_private_key(
                        fi.read(),
                        password=None,
                        backend=default_backend()
                    )
                # 如果未抛异常，说明证书未加密
                return False
            except TypeError:
                # 抛出异常说明证书已经加密
                return True
            except Exception:
                # 其他异常表示key文件不是pem格式或者已经损坏
                logger.error("an exception occurred while checking the key file whether encrypted or not")
                return False

        if not check(key_path):
            raise ValueError("key file must encrypted")

    def post(self, url: str, body: str, headers: Dict):
        if not is_url_valid(url, self.use_http):
            logger.error("url check failed")
            return Result(False, "")

        try:
            response = self.pool.request(method='POST',
                                         url=url,
                                         body=body,
                                         headers=headers,
                                         preload_content=False)
        except urllib3_TimeoutError:
            logger.error("The request timed out")
            return Result(False, "")
        except HTTPError:
            logger.error("Request failed due to HTTP error")
            return Result(False, "")
        except Exception:
            logger.error("request failed")
            return Result(False, "")

        try:
            content_length = int(response.headers.get("Content-Length"))
        except ValueError as e:
            logger.error(f"Invalid Content-Length header in response: {e}")
            return Result(False, "")
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
                logger.error(f"Failed to read response: {e}")
                return Result(False, "")

            return Result(True, response_data)
        else:
            logger.error(f"request failed with status code {response.status}")
            return Result(False, "")

    def post_streamly(self, url: str, body: str, headers: Dict, chunk_size: int = 1024):
        if not is_url_valid(url, self.use_http):
            logger.error("url check failed")
            yield Result(False, "")

        try:
            response = self.pool.request(method='POST', url=url, body=body, headers=headers, preload_content=False)
        except urllib3_TimeoutError:
            logger.error("The request timed out")
            yield Result(False, "")
            return
        except HTTPError:
            logger.error("Request failed due to HTTP error")
            yield Result(False, "")
            return
        except Exception:
            logger.error(f"request failed")
            yield Result(False, "")
            return

        try:
            content_type = response.headers.get("Content-Type")
            if content_type is None:
                raise ValueError("Invalid Content-Type header")
            content_type = str(content_type)
        except KeyError as e:
            logger.error(f"Content-Type header is missing: {e}")
            yield Result(False, "")
            return
        except ValueError as e:
            logger.error(f"Invalid Content-Type header: {e}")
            yield Result(False, "")
            return
        except Exception as e:
            logger.error(f"Failed to get Content-Type, unexpected error: {e}")
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
        except urllib3.exceptions.HTTPError as e:
            logger.error(f"HTTP error while reading response: {e}")
            yield Result(False, "")
        except Exception as e:
            logger.error(f"read response failed, find exception: {e}")
            yield Result(False, "")

    def _check_https_para(self, client_param: ClientParam):
        SecFileCheck(client_param.ca_file, MAX_CERT_FILE_SIZE).check()
        self._check_ca_content(client_param.ca_file)

        # crt key pwd 3个参数须同时有效
        if client_param.crt_file or client_param.key_file or client_param.pwd:
            SecFileCheck(client_param.crt_file, MAX_CERT_FILE_SIZE).check()
            SecFileCheck(client_param.key_file, MAX_CERT_FILE_SIZE).check()
            self._check_key_file_whether_encrypted(client_param.key_file)
            self._check_password(client_param.pwd)

        if client_param.crl_file:
            SecFileCheck(client_param.crl_file, MAX_CERT_FILE_SIZE).check()
