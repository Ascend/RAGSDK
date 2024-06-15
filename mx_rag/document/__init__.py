# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

__all__ = [
    "loader",
    "splitter",
    "parse_file",
    "SUPPORT_IMAGE_TYPE",
    "SUPPORT_DOC_TYPE"
]

from mx_rag.document import loader
from mx_rag.document import splitter
from mx_rag.document.parser import parse_file, SUPPORT_IMAGE_TYPE, SUPPORT_DOC_TYPE
