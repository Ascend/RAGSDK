# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from abc import ABC, abstractmethod
from typing import List

from pydantic.v1 import BaseModel, Field, validator

from mx_rag.utils.common import validate_sequence, MAX_PAGE_CONTENT


class StorageError(Exception):
    pass


class MxDocument(BaseModel):
    page_content: str = Field(max_length=MAX_PAGE_CONTENT)
    metadata: dict = Field(default_factory=dict)
    document_name: str = Field(max_length=1024)

    class Config:
        arbitrary_types_allowed = True

    @validator('metadata')
    def _validate_metadata(cls, metadata):
        if not validate_sequence(metadata):
            raise ValueError("check MxDocument metadata failed")

        return metadata


class Docstore(ABC):
    @abstractmethod
    def search(self, chunk_id) -> MxDocument:
        pass

    @abstractmethod
    def delete(self, document_id):
        pass

    @abstractmethod
    def add(self, documents, document_id):
        pass

    @abstractmethod
    def get_all_chunk_id(self) -> List[int]:
        pass

    @abstractmethod
    def get_all_document_id(self) -> List[int]:
        pass

    @abstractmethod
    def search_by_document_id(self, document_id: int):
        pass

    @abstractmethod
    def update(self, chunk_ids: List[int], texts: List[str]):
        pass
