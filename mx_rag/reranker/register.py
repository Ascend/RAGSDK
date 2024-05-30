# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from enum import Enum
from typing import Dict

from .local import LocalReranker
from .service import TEIReranker


class RerankerModelType(Enum):
    LOCAL = 1
    TEI = 2


TYPE_TO_RERANKER_MODEL: Dict = {
    RerankerModelType.LOCAL.value: LocalReranker,
    RerankerModelType.TEI.value: TEIReranker
}
