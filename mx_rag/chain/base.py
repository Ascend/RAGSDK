# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from abc import ABC, abstractmethod
from typing import Union, Iterator, Dict

from mx_rag.llm.llm_parameter import LLMParameterConfig


class Chain(ABC):
    @abstractmethod
    def query(self, text: str, llm_config: LLMParameterConfig = LLMParameterConfig(), *args, **kwargs) \
            -> Union[Dict, Iterator[Dict]]:
        """ query by text"""
