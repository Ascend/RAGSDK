# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import Dict

from langchain_core.retrievers import BaseRetriever

from mx_rag.utils.common import validate_params
from mx_rag.llm.llm_parameter import LLMParameterConfig
from mx_rag.chain.base import Chain
from mx_rag.llm import Text2ImgMultiModel


class Text2ImgChain(Chain):
    """ 大模型输入prompt，生成prompt相关的图片 """

    @validate_params(
        multi_model=dict(validator=lambda x: isinstance(x, Text2ImgMultiModel))
    )
    def __init__(self, multi_model):
        self._multi_model = multi_model

    def query(self, text: str, llm_config: LLMParameterConfig = LLMParameterConfig(), *args, **kwargs) -> Dict:
        return self._multi_model.text2img(prompt=kwargs.get("prompt"),
                                          output_format=kwargs.get("output_format", "png"),
                                          size=kwargs.get("size", "512*512"))
