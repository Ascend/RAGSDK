# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from abc import ABC, abstractmethod

from pydantic.v1 import BaseModel, Field, validator


class StorageError(Exception):
    pass


class MxDocument(BaseModel):
    page_content: str = Field(min_length=1, max_length=1024)
    metadata: dict = Field(default_factory=dict)
    document_name: str = Field(min_length=1, max_length=1024)

    class Config:
        arbitrary_types_allowed = True

    @validator('metadata')
    def _validate_metadata(cls, metadata):
        for k, v in metadata.items():
            if isinstance(k, str) and isinstance(v, str):
                continue
            raise ValueError("metadata must be type Dict[str, str]")
        if len(str(metadata)) > 1024:
            raise ValueError("The length of the MxDocument metadata converted into a str cannot exceed 1024.")
        return metadata


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
