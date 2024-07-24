# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

__all__ = [
    "SQLiteDocstore",
    "Docstore"
]

from .base_storage import Docstore
from .sqlite_storage import SQLiteDocstore
