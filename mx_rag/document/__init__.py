# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

__all__ = [
    "Doc",
    "loader",
    "splitter",
    "SUPPORT_IMAGE_TYPE",
    "SUPPORT_DOC_TYPE"
]

from mx_rag.document import loader
from mx_rag.document import splitter
from mx_rag.document.doc import Doc


SUPPORT_IMAGE_TYPE = (".jpg", ".png")
SUPPORT_DOC_TYPE = (".docx", ".xlsx", ".xls", ".csv", ".pdf")