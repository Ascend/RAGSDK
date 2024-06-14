# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from loguru import logger

from mx_rag.chain.base import Chain


class Text2ImgChain(Chain):
    """ 大模型输入prompt，生成prompt相关的图片 """

    def __init__(self, multi_model, retriever=None):
        self._multi_model = multi_model
        self._retriever = retriever

    def query(self, text : str, *args, **kwargs) -> str:
        if "prompt" not in kwargs:
            logger.error("input param must contain prompt")
            return ""

        return self._multi_model.text2img(kwargs["prompt"], kwargs.get("output_format", "png"))