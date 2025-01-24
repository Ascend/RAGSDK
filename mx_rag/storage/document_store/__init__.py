# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

__all__ = [
    "SQLiteDocstore",
    "MilvusDocstore",
    "Docstore",
    "MxDocument"
]

from .base_storage import Docstore, MxDocument
from .sqlite_storage import SQLiteDocstore
from .milvus_storage import MilvusDocstore
