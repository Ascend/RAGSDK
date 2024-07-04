# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from datetime import datetime
from enum import Enum

from OpenSSL import crypto


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
