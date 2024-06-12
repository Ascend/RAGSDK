# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import Union, Iterator

from loguru import logger

from mx_rag.chain import Chain
from mx_rag.chain.router.textclassifier import TextClassifier


class QueryRouter:
    """ 解析输入文本进行分类，调用对应的chain进行大模型推理 """

    LABELS = ["text generate text", "text generate image", "image generate image"]

    def __init__(self, classifier: TextClassifier):
        self._classifier = classifier
        self._chains: dict = {}
        self._labels: list = []

    def register_chain(self, label: str, chain: Chain):
        """ 根据文本分类标签，注册大模型chain """
        if label not in self.LABELS:
            logger.error(f"register [{label}] chain failed, because not in {self.LABELS}")
            return

        if label in self._chains.keys() and isinstance(self._chains[label], Chain):
            logger.warning(f"[{label}] chain has been registered")
            return

        self._chains[label] = chain
        self._labels.append(label)

    def get_register_labels(self) -> list[str]:
        return self._labels

    def unregister_chain(self, label: str):
        """ 根据文本分类标签，卸载大模型chain """
        if label not in self._chains.keys():
            logger.warning(f"[{label}] chain has not been registered")
            return

        del(self._chains[label])
        self._labels.remove(label)

    def route_to_llm(self, text: str, *args, **kwargs) -> Union[str, Iterator]:
        """ 解析text 分类标签，调用对应的大模型chain """
        purpose = self._parse_purpose(text)
        data = ""
        if purpose not in self._chains.keys():
            logger.error("classify query purpose failed")
            return data

        chain = self._chains[purpose]
        return chain.query(text, *args, **kwargs)

    def _parse_purpose(self, text: str) -> str:
        return self._classifier.classify(text, self._labels)

