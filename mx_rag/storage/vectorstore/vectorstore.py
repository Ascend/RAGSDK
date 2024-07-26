# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from abc import ABC, abstractmethod


class VectorStore(ABC):

    @abstractmethod
    def delete(self, ids):
        pass

    @abstractmethod
    def search(self, embeddings, k):
        pass

    @abstractmethod
    def add(self, embeddings, ids):
        pass

    def save_local(self):
        pass

    def get_save_file(self):
        return ""

    def get_ntotal(self) -> int:
        return 0
