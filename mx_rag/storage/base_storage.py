# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import inspect
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field


class StorageError(Exception):
    def __init__(self, err_msg: str):
        self.err_msg = err_msg
        info: inspect.FrameInfo = inspect.stack()[1]
        msg = f"{info.function}({Path(info.filename).name}:{info.lineno})-{err_msg}"
        super().__init__(msg)


class Document(BaseModel):
    page_content: str
    metadata: dict = Field(default_factory=dict)
    document_name: str


class Docstore(ABC):
    @abstractmethod
    def search(self, *args, **kwargs) -> Document:
        pass

    @abstractmethod
    def delete(self, *args, **kwargs):
        pass

    @abstractmethod
    def add(self, *args, **kwargs):
        pass

    @abstractmethod
    def check_document_exist(self, doc_name: str) -> bool:
        pass
