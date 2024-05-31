# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    MAX_SIZE_MB = 100
    MAX_PAGE_NUM = 1000

    def __init__(self, file_path):
        self.file_path = file_path

    @abstractmethod
    def load(self):
        pass
