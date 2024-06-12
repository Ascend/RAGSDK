# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from enum import Enum
from typing import Dict

from .local import TextEmbedding, ImageEmbedding
from .service import TEIEmbedding


class EmbeddingModelType(Enum):
    LOCAL_TEXT = 1
    LOCAL_IMAGE = 2
    TEI = 3


TYPE_TO_EMBEDDING_MODEL: Dict = {
    EmbeddingModelType.LOCAL_TEXT.value: TextEmbedding,
    EmbeddingModelType.LOCAL_IMAGE.value: ImageEmbedding,
    EmbeddingModelType.TEI.value: TEIEmbedding
}
