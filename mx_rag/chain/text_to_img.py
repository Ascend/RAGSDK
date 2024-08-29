# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import Dict

from loguru import logger

from langchain_core.retrievers import BaseRetriever

from mx_rag.utils.common import validate_params
from mx_rag.chain.base import Chain
from mx_rag.llm import Text2ImgMultiModel


class Text2ImgChain(Chain):
    """ 大模型输入prompt，生成prompt相关的图片 """

    @validate_params(
        multi_model=dict(validator=lambda x: isinstance(x, Text2ImgMultiModel)),
        retriever=dict(validator=lambda x: isinstance(x, BaseRetriever) or x is None)
    )
    def __init__(self, multi_model, retriever=None):
        self._multi_model = multi_model
        self._retriever = retriever

    def query(self, text : str, *args, **kwargs) -> Dict:
        if "prompt" not in kwargs:
            logger.error("input param must contain prompt")
            return {}

        return self._multi_model.text2img(kwargs["prompt"], kwargs.get("output_format", "png"))