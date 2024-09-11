# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import os.path
import ssl
from typing import Union

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
    def get_cipher_suites():
        return ':'.join(SAFE_CIPHER_SUITES)

    @staticmethod
    def get_pwd_callback(pwd):
        if pwd is None:
            return pwd

        def pwd_callback():
            return pwd

        return pwd_callback

    @staticmethod
    def enable_crl_check(ctx):
        ctx.verify_flags |= ssl.VERIFY_CRL_CHECK_LEAF

    @staticmethod
    def disable_crl_check(ctx):
        ctx.verify_flags &= ~ssl.VERIFY_CRL_CHECK_LEAF

    @staticmethod
    def get_ssl_context(cafile, certfile, keyfile, pwd):
        """
        获取ssl.context
        cafile:   根证书文件，用于校验对端
        certfile: 证书文件
        keyfile:  私钥文件
        pwd:      私钥文件的口令。可以传入明文、密文、或者装有密文的文件
        """
        context = ssl.SSLContext()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        context.set_ciphers(':'.join(SAFE_CIPHER_SUITES))
        try:
            # 如果提供CA文件，则加载
            if cafile is not None:
                if not os.path.exists(cafile):
                    raise FileNotFoundError(f"CA file '{cafile}' not found")
                context.load_verify_locations(cafile)
            # 加载certificate和key文件
            if not os.path.exists(certfile):
                raise FileNotFoundError(f"Certificate file '{certfile}' not found.")
            if not os.path.exists(keyfile):
                raise FileNotFoundError(f"Key file '{keyfile}' not found.")
            context.load_cert_chain(certfile, keyfile, password=TlsConfig.get_pwd_callback(pwd))
            return True, context
        except FileNotFoundError as fnf_error:
            return False, f"File error: {fnf_error}"
        except ssl.SSLError as ssl_error:
            return False, f"SSL error: {ssl_error}"
        except Exception as e:
            return False, f"Unexpected error while setting up SSL context: {e}"

    @staticmethod
    def get_init_context():
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        context.set_ciphers(':'.join(SAFE_CIPHER_SUITES))
        context.verify_mode = ssl.CERT_REQUIRED
        return context

    @classmethod
    def get_client_context_with_cadata(cls, ca_data: Union[str, bytes], crl_data: Union[str, bytes] = None):
        """
        利用证书内容和吊销列表内容，获取ssl.context, 通常客户端使用
        使用tls1.2和tls1.3
        ca_data:  证书内容，用于校验对端
        crl_data; 吊销列表内容，用于校验对端是否被吊销
        """
        context = cls.get_init_context()
        try:
            context.load_verify_locations(cadata=ca_data)
            if crl_data:
                context.load_verify_locations(cadata=crl_data)
                cls.enable_crl_check(context)

            return True, context
        except ssl.SSLError as ssl_error:
            return False, f"SSL error occurred: {ssl_error}"
        except TypeError as type_error:
            return False, f"Invalid type for CA or CRL data: {type_error}"
        except Exception as error_info:
            return False, f"An unexpected error occurred while setting up the SSL context: {error_info}"

    @classmethod
    def get_client_ssl_context(cls, ca_file: str, crl_file: str = None):
        """
        load 单个证书路径，到ssl.context, 获取context，通常客户端使用
        使用tls1.2和tls1.3
        ca_file:   根证书文件路径，用于校验对端
        crl_file:  吊销列表文件路径，用于校验对端是否被吊销
        """
        context = cls.get_init_context()
        try:
            context.load_verify_locations(ca_file)
            if crl_file:
                context.load_verify_locations(crl_file)
                cls.enable_crl_check(context)

            return True, context
        except FileNotFoundError as fnf_error:
            # 处理CA或CRL文件找不到的情况
            return False, f"File not found: {fnf_error}"
        except PermissionError as perm_error:
            # 处理CA或CRL文件权限相关的问题
            return False, f"Permission denied: {perm_error}"
        except ssl.SSLError as ssl_error:
            # 处理SSL相关的错误 (比如, 无效的证书格式, SSL设置失败)
            return False, f"SSL error occurred: {ssl_error}"
        except Exception as error_info:
            return False, f"An unexpected error occurred while setting up the SSL context: {error_info}"
