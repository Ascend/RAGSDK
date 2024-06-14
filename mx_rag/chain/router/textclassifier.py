# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from abc import ABC, abstractmethod

from loguru import logger
from transformers import pipeline

from mx_rag.utils import FileCheck


class TextClassifier(ABC):

    @abstractmethod
    def classify(self, text: str, labels: list[str]) -> str:
        """ classify labels """


class ZeroShotTextClassifier(TextClassifier):
    def __init__(self, model_path : str, dev_id: int = 0):
        FileCheck.check_path_is_exist_and_valid(model_path)

        self._classifier = pipeline("zero-shot-classification",
                                     model=model_path,
                                     device_map=f'npu:{dev_id}')

    def classify(self, text: str, labels: list[str]) -> str:
        """ 根据labels 集合，分类text匹配的label """
        hypothesis_template = "This text is about {}"
        class_res = self._classifier(text, labels, hypothesis_template=hypothesis_template, multi_label=False)
        logger.info(f"classify input text result{class_res}")
        return class_res["labels"][0]
