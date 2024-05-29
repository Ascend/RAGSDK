# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from enum import Enum
from typing import Dict

from .local import LocalEmbedding
from .service import TEIEmbedding


class EmbeddingModelType(Enum):
    LOCAL = 1
    TEI = 2


TYPE_TO_EMBEDDING_MODEL: Dict = {
    EmbeddingModelType.LOCAL: LocalEmbedding,
    EmbeddingModelType.TEI: TEIEmbedding
}
