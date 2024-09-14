# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
from typing import Dict, Any, List, Tuple, Type, Optional
from dataclasses import dataclass
from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters.base import Language, TextSplitter

from mx_rag.utils.common import validate_params


@dataclass
class LoaderInfo:
    loader_class: Type
    loader_params: Dict[str, Any]


@dataclass
class SplitterInfo:
    splitter_class: Type
    splitter_params: Dict[str, Any]


class LoaderMng:
    MAX_REGISTER_LOADER_NUM = 1000
    MAX_REGISTER_SPLITTER_NUM = 1000
    IMAGE_TYPE = (".jpg", ".png")

    def __init__(self):
        self.loaders: Dict[Type, Tuple[List[str], LoaderInfo]] = {}
        self.splitters: Dict[Type, Tuple[List[str], SplitterInfo]] = {}

    @validate_params(
        file_types=dict(validator=lambda x: all(isinstance(item, str) for item in x) and 0 <= len(x) <= 32)
    )
    def register_loader(self, loader_class: BaseLoader, file_types: List[str],
                        loader_params: Optional[Dict[str, Any]] = None):
        if len(self.loaders) >= self.MAX_REGISTER_LOADER_NUM:
            raise ValueError(f"More than {self.MAX_REGISTER_LOADER_NUM} loaders are registered")
        self.loaders[loader_class] = (file_types, LoaderInfo(loader_class, loader_params or {}))

    @validate_params(
        file_types=dict(validator=lambda x: all(isinstance(item, str) for item in x) and 0 <= len(x) <= 32)
    )
    def register_splitter(self, splitter_class: TextSplitter, file_types: List[str],
                          splitter_params: Optional[Dict[str, Any]] = None):
        if len(self.splitters) >= self.MAX_REGISTER_SPLITTER_NUM:
            raise ValueError(f"More than {self.MAX_REGISTER_SPLITTER_NUM} splitters are registered")
        if bool(set(self.IMAGE_TYPE) & set(file_types)):
            raise KeyError(f"Unsupported register splitter for file type {set(self.IMAGE_TYPE) & set(file_types)}")
        self.splitters[splitter_class] = (file_types, SplitterInfo(splitter_class, splitter_params or {}))

    def get_loader(self, file_suffix: str) -> LoaderInfo:
        for file_types, loader_info in self.loaders.values():
            if file_suffix in file_types:
                return loader_info
        raise KeyError(f"No loader registered for file type '{file_suffix}'")

    def get_splitter(self, file_suffix: str) -> SplitterInfo:
        for file_types, splitter_info in self.splitters.values():
            if file_suffix in file_types:
                return splitter_info
            elif file_suffix in self.IMAGE_TYPE:
                return None
        raise KeyError(f"No splitter registered for file type '{file_suffix}'")

    def unregister_loader(self, loader_class: Type):
        if loader_class in self.loaders:
            del self.loaders[loader_class]
        else:
            raise KeyError(f"Loader class '{loader_class}' is not registered")

    def unregister_splitter(self, splitter_class: Type):
        if splitter_class in self.splitters:
            del self.splitters[splitter_class]
        else:
            raise KeyError(f"Splitter class '{splitter_class}' is not registered")
