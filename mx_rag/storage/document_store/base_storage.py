# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import inspect
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field


class StorageError(Exception):
    pass


class MxDocument(BaseModel):
    page_content: str
    metadata: dict = Field(default_factory=dict)
    document_name: str


class Docstore(ABC):
    @abstractmethod
    def search(self, index_id) -> MxDocument:
        pass

    @abstractmethod
    def delete(self, doc_name):
        pass

    @abstractmethod
    def add(self, documents):
        pass
