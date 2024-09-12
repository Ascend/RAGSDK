# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import functools
import inspect
import re
from datetime import datetime
from enum import Enum
from typing import List

from OpenSSL import crypto
from loguru import logger

FILE_COUNT_MAX = 8000
INT_32_MAX = 2 ** 31 - 1
MAX_DEVICE_ID = 63
MAX_TOP_K = 10000
MAX_QUERY_LENGTH = 128 * 1024 * 1024
EMBBEDDING_TEXT_COUNT = 1000 * 1000
EMBBEDDING_IMG_COUNT = 1000
IMG_EMBBEDDING_TEXT_LEN = 256
MAX_FILE_SIZE = 100 * 1024 * 1024
TEXT_MAX_LEN = 1000 * 1000
MAX_VEC_DIM = 1024 * 1024
NODE_MAX_TEXT_LENGTH = 128 * 1024 * 1024
# tree中最大节点数
TREE_MAX_NODES = 2 ** 31 - 1
# tree中最大层数
TREE_MAX_LAYERS = 2 ** 31 - 1
MILVUS_INDEX_TYPES = ["FLAT"]
MILVUS_METRIC_TYPES = ["L2", "IP", "COSINE"]

MAX_PROMPT_LENGTH = 1 * 1024 * 1024
MAX_URL_LENGTH = 128
MAX_MODEL_NAME_LENGTH = 128

KB = 1024
MB = 1048576  # 1024 * 1024
GB = 1073741824  # 1024 * 1024 * 1024


class UrlUtilException(Exception):
    pass


def safe_get(data, keys, default=None):
    """
    安全地获取嵌套字典或列表中的值。

    :param data: 字典或列表数据
    :param keys: 键或索引列表，表示嵌套层级
    :param default: 如果键或索引不存在，返回的默认值
    :return: 对应键或索引的值或默认值
    """
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        elif isinstance(data, list) and isinstance(key, int) and 0 <= key < len(data):
            data = data[key]
        else:
            return default
    return data


def _get_value_from_param(arg_name, func, *args, **kwargs):
    sig = inspect.signature(func)
    # 从传入参数中获取要校验的value
    for param_name, param in sig.bind(*args, **kwargs).arguments.items():
        if arg_name == param_name:
            return param
    # 传入参数中没有则从方法定义中取默认值
    for name, param in sig.parameters.items():
        if arg_name == name:
            return param.default
    # 都没有抛出异常
    raise ValueError(f"Required parameter '{arg_name}' of function {func.__name__} is missing.")


def validate_params(**validators):
    """
    定义一个装饰器，用于验证函数的多个参数。在方法上使用注释
    @validate_params(
        name=dict(validator=lambda x: isinstance(x, str)),
        age=dict(validator=lambda x: 10 <= x <= 30)
    )
    :param validators: 一个包含验证函数的字典，每个函数用于验证一个特定的参数。
    :return: 装饰器函数
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 对每个参数应用验证函数
            for arg_name, validator in validators.items():
                # 检查是否通过位置或关键字传递了参数
                value = _get_value_from_param(arg_name, func, *args, **kwargs)
                # 运行验证函数
                if not validator['validator'](value):
                    raise ValueError(f"The parameter '{arg_name}' of function '{func.__name__}' "
                                     f"is invalid, please see the document.")
            # 如果所有参数都通过验证，则调用原始函数
            return func(*args, **kwargs)

        return wrapper

    return decorator


class PubkeyType(Enum):
    EVP_PKEY_RSA = 6
    EVP_PKEY_DSA = 116
    EVP_PKEY_DH = 28
    EVP_PKEY_EC = 408


class ParseCertInfo:
    """解析根证书信息类"""

    def __init__(self, cert_buffer: str):
        if not cert_buffer:
            raise ValueError("Cert buffer is null.")

        self.cert_info = crypto.load_certificate(crypto.FILETYPE_PEM, str.encode(cert_buffer))
        self.serial_num = hex(self.cert_info.get_serial_number())[2:].upper()
        self.subject_components = self.cert_info.get_subject().get_components()
        self.issuer_components = self.cert_info.get_issuer().get_components()
        self.fingerprint = self.cert_info.digest("sha256").decode()
        self.start_time = datetime.strptime(self.cert_info.get_notBefore().decode(), '%Y%m%d%H%M%SZ')
        self.end_time = datetime.strptime(self.cert_info.get_notAfter().decode(), '%Y%m%d%H%M%SZ')
        self.signature_algorithm = self.cert_info.get_signature_algorithm().decode()
        self.signature_len = self.cert_info.get_pubkey().bits()
        self.cert_version = self.cert_info.get_version() + 1
        self.pubkey_type = self.cert_info.get_pubkey().type()
        self.ca_pub_key = self.cert_info.get_pubkey().to_cryptography_key()
        self.extensions = {}
        for i in range(self.cert_info.get_extension_count()):
            ext = self.cert_info.get_extension(i)
            ext_name = ext.get_short_name().decode()
            try:
                self.extensions[ext_name] = str(ext)
            except (TypeError, ValueError) as e:
                logger.warning(f"Type error or value error, format {ext_name}: {e}")
                continue
            except Exception as e:
                logger.warning(f"format '{ext_name}' str info in certificate failed: {e}")
                continue

    @property
    def subject(self) -> str:
        return ", ".join([f"{item[0].decode()}={item[1].decode()}" for item in self.subject_components])

    @property
    def issuer(self) -> str:
        return ", ".join([f"{item[0].decode()}={item[1].decode()}" for item in self.issuer_components])

    def to_dict(self) -> dict:
        return {
            "SerialNum": self.serial_num,
            "Subject": self.subject,
            "Issuer": self.issuer,
            "Fingerprint": self.fingerprint,
            "Date": f"{self.start_time}--{self.end_time}",
        }


def validata_list_str(texts: List[str], length_limit: List[int], str_limit: List[int]):
    """
    用于List[str]类型的数据校验
    Args:
        texts: 输入数据字符串列表
        length_limit: 列表长度范围
        str_limit: 字符串长度范围

    Returns:

    """
    min_length_limit = length_limit[0]
    max_length_limit = length_limit[1]
    min_str_limit = str_limit[0]
    max_str_limit = str_limit[1]
    if not min_length_limit <= len(texts) <= max_length_limit:
        logger.error(f"The List[str] length not in [{min_length_limit}, {max_length_limit}]")
        return False
    for text in texts:
        if not isinstance(text, str):
            logger.error("The element in the list is not a string.")
            return False
        if not min_str_limit <= len(text) <= max_str_limit:
            logger.error(f"The element in List[str] length not in [{min_str_limit}, {max_str_limit}]")
            return False
    return True
