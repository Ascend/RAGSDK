# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import ssl

from glib.security import TlsConfig
from mx_rag.utils import ClientParam

SAFE_CIPHER_SUITES = [
    'ECDHE-ECDSA-AES128-GCM-SHA256',
    'ECDHE-ECDSA-AES256-GCM-SHA384',
    "ECDHE-ECDSA-CHACHA20-POLY1305-SHA256",
    'ECDHE-RSA-AES128-GCM-SHA256',
    'ECDHE-RSA-AES256-GCM-SHA384',
    "ECDHE-RSA-CHACHA20-POLY1305-SHA256"
]


def get_two_way_auth_ssl_context(client_param: ClientParam) -> ssl.SSLContext:
    success, ssl_ctx = TlsConfig.get_ssl_context(client_param.ca_file, client_param.crt_file,
                                                 client_param.key_file, client_param.pwd)
    if not success:
        raise ValueError('unable to add ca or crt or key file to ssl context')

    if client_param.crl_file:
        ssl_ctx.load_verify_locations(client_param.crl_file)
        TlsConfig.enable_crl_check(ssl_ctx)

    ssl_ctx.verify_mode = ssl.CERT_REQUIRED
    return ssl_ctx


def get_one_way_auth_ssl_context(client_param: ClientParam) -> ssl.SSLContext:
    success, ssl_ctx = TlsConfig.get_client_ssl_context(client_param.ca_file, client_param.crl_file)
    if not success:
        raise ValueError('unable to add ca_file for request')

    ssl_ctx.verify_mode = ssl.CERT_REQUIRED
    return ssl_ctx


def get_default_context():
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    context.set_ciphers(':'.join(SAFE_CIPHER_SUITES))
    context.verify_mode = ssl.CERT_REQUIRED
    return context
