# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import ssl

SAFE_CIPHER_SUITES = [
    'ECDHE-ECDSA-AES128-GCM-SHA256',
    'ECDHE-ECDSA-AES256-GCM-SHA384',
    "ECDHE-ECDSA-CHACHA20-POLY1305-SHA256",
    'ECDHE-RSA-AES128-GCM-SHA256',
    'ECDHE-RSA-AES256-GCM-SHA384',
    "ECDHE-RSA-CHACHA20-POLY1305-SHA256"
]


class TlsConfig(object):
    @staticmethod
    def enable_crl_check(ctx):
        ctx.verify_flags |= ssl.VERIFY_CRL_CHECK_CHAIN

    @classmethod
    def get_client_ssl_context(cls, ca_file: str, crl_file: str = None):
        """
        load 单个证书路径，到ssl.context, 获取context，通常客户端使用
        使用tls1.2和tls1.3
        ca_file:   根证书文件路径，用于校验对端
        crl_file:  吊销列表文件路径，用于校验对端是否被吊销
        """
        context = cls._get_init_context()
        try:
            context.load_verify_locations(ca_file)
            if crl_file:
                context.load_verify_locations(crl_file)
                cls.enable_crl_check(context)

            return True, context
        except Exception as error_info:
            return False, f"get client ssl context failed: {error_info}"

    @staticmethod
    def _get_init_context():
        context = ssl.create_default_context()
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.set_ciphers(':'.join(SAFE_CIPHER_SUITES))
        context.verify_mode = ssl.CERT_REQUIRED
        return context
