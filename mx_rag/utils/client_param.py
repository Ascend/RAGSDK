# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import ssl

from mx_rag.utils.common import validate_params, BOOL_TYPE_CHECK_TIP, STR_TYPE_CHECK_TIP, MB


class ClientParam:
    @validate_params(
        use_http=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP),
        ca_file=dict(validator=lambda x: isinstance(x, str), message=STR_TYPE_CHECK_TIP),
        crt_file=dict(validator=lambda x: isinstance(x, str), message=STR_TYPE_CHECK_TIP),
        key_file=dict(validator=lambda x: isinstance(x, str), message=STR_TYPE_CHECK_TIP),
        crl_file=dict(validator=lambda x: isinstance(x, str), message=STR_TYPE_CHECK_TIP),
        pwd=dict(validator=lambda x: isinstance(x, str), message=STR_TYPE_CHECK_TIP),
        timeout=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 600,
                     message="param must be int and value range (0, 600]"),
        response_limit_size=dict(validator=lambda x: isinstance(x, int) and 0 < x <= 10 * MB,
                                 message="param must be int and value range (0, 10MB]"),
    )
    def __init__(self,
                 use_http: bool = False,
                 ca_file: str = "",
                 crt_file: str = "",
                 key_file: str = "",
                 crl_file: str = "",
                 pwd: str = "",
                 timeout: int = 60,
                 response_limit_size: int = MB):
        self.use_http = use_http
        self.ca_file = ca_file
        self.crt_file = crt_file
        self.key_file = key_file
        self.crl_file = crl_file
        self.pwd = pwd
        self.timeout: int = timeout
        self.response_limit_size: int = response_limit_size
