# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from abc import ABC, abstractmethod


class VectorStore(ABC):

    @abstractmethod
    def delete(self, *args, **kwargs):
        pass

    @abstractmethod
    def search(self, *args, **kwargs):
        pass

    @abstractmethod
    def add(self, *args, **kwargs):
        pass
