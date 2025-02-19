# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
__all__ = [
    "TextEmbedding",
    "ImageEmbedding",
    "SparseEmbedding"
]

from mx_rag.embedding.local.text_embedding import TextEmbedding
from mx_rag.embedding.local.img_embedding import ImageEmbedding
from mx_rag.embedding.local.sparse_embedding import SparseEmbedding
