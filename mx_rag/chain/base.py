# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from abc import ABC, abstractmethod
from typing import Union, Iterator, Dict


class Chain(ABC):
    @abstractmethod
    def query(self, text : str, *args, **kwargs) -> Union[Dict, Iterator[Dict]]:
        """ query by text"""

