# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from dataclasses import dataclass, field
from enum import Enum


class KgOprMode(Enum):
    NEW = 1


@dataclass
class GraphUpdatedData:
    added_nodes: list = field(default_factory=list)
    added_edges: list = field(default_factory=list)
    deleted_nodes: list = field(default_factory=list)
    deleted_edges: list = field(default_factory=list)