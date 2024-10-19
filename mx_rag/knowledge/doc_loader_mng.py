# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import Dict, Any, List, Tuple, Type, Optional
from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters.base import TextSplitter

from mx_rag.utils.common import (DICT_TYPE_CHECK_TIP, validata_list_str, validate_params, NO_SPLIT_FILE_TYPE,
                                 FILE_TYPE_COUNT, CLASS_TYPE_CHECK_TIP, STR_TYPE_CHECK_TIP_1024)


class LoaderInfo:
    def __init__(self,
                 loader_class: Type,
                 loader_params: Dict[str, Any]):
        self.loader_class = loader_class
        self.loader_params = loader_params


class SplitterInfo:
    def __init__(self,
                 splitter_class: Type,
                 splitter_params: Dict[str, Any]):
        self.splitter_class = splitter_class
        self.splitter_params = splitter_params


class LoaderMng:
    MAX_REGISTER_LOADER_NUM = 1000
    MAX_REGISTER_SPLITTER_NUM = 1000

    def __init__(self):
        self.loaders: Dict[Type, Tuple[List[str], LoaderInfo]] = {}
        self.loader_types: list = []
        self.splitters: Dict[Type, Tuple[List[str], SplitterInfo]] = {}
        self.splitter_types: list = []

    @validate_params(
        loader_class=dict(validator=lambda x: isinstance(x, type), message=CLASS_TYPE_CHECK_TIP),
        file_types=dict(validator=lambda x: validata_list_str(x, [1, FILE_TYPE_COUNT], [1, FILE_TYPE_COUNT]),
                        message="param must meets: Type is List[str], "
                                "list length range [1, 32], str length range [1, 32]"),
        loader_params=dict(validator=lambda x: isinstance(x, Dict) or x is None, message=DICT_TYPE_CHECK_TIP)
    )
    def register_loader(self, loader_class: BaseLoader, file_types: List[str],
                        loader_params: Optional[Dict[str, Any]] = None):
        if len(self.loaders) >= self.MAX_REGISTER_LOADER_NUM:
            raise ValueError(f"More than {self.MAX_REGISTER_LOADER_NUM} loaders are registered")
        repeat_type = list(set(self.loader_types) & set(file_types))
        if len(repeat_type) > 0:
            raise ValueError(f"loaders type {repeat_type} has been registered")
        self.loader_types.extend(file_types)
        self.loaders[loader_class] = (file_types, LoaderInfo(loader_class, loader_params or {}))

    @validate_params(
        splitter_class=dict(validator=lambda x: isinstance(x, type), message=CLASS_TYPE_CHECK_TIP),
        file_types=dict(validator=lambda x: validata_list_str(x, [1, FILE_TYPE_COUNT], [1, FILE_TYPE_COUNT]),
                        message="param must meets: Type is List[str], "
                                "list length range [1, 32], str length range [1, 32]"),

        splitter_params=dict(validator=lambda x: isinstance(x, Dict) or x is None, message=DICT_TYPE_CHECK_TIP)
    )
    def register_splitter(self, splitter_class: TextSplitter, file_types: List[str],
                          splitter_params: Optional[Dict[str, Any]] = None):
        if len(self.splitters) >= self.MAX_REGISTER_SPLITTER_NUM:
            raise ValueError(f"More than {self.MAX_REGISTER_SPLITTER_NUM} splitters are registered")
        if bool(set(NO_SPLIT_FILE_TYPE) & set(file_types)):
            raise KeyError(f"Unsupported register splitter for file type {set(NO_SPLIT_FILE_TYPE) & set(file_types)}")
        repeat_type = list(set(self.splitter_types) & set(file_types))
        if len(repeat_type) > 0:
            raise ValueError(f"splitters type {repeat_type} has been registered")
        self.splitter_types.extend(file_types)
        self.splitters[splitter_class] = (file_types, SplitterInfo(splitter_class, splitter_params or {}))

    @validate_params(
        file_suffix=dict(validator=lambda x: isinstance(x, str) and len(x) < 1024, message=STR_TYPE_CHECK_TIP_1024))
    def get_loader(self, file_suffix: str) -> LoaderInfo:
        for file_types, loader_info in self.loaders.values():
            if file_suffix in file_types:
                return loader_info
        raise KeyError(f"No loader registered for file type '{file_suffix}'")

    @validate_params(
        file_suffix=dict(validator=lambda x: isinstance(x, str) and len(x) < 1024, message=STR_TYPE_CHECK_TIP_1024))
    def get_splitter(self, file_suffix: str) -> SplitterInfo:
        for file_types, splitter_info in self.splitters.values():
            if file_suffix in file_types:
                return splitter_info

        raise KeyError(f"No splitter registered for file type '{file_suffix}'")

    @validate_params(
        loader_class=dict(validator=lambda x: isinstance(x, type), message=CLASS_TYPE_CHECK_TIP))
    def unregister_loader(self, loader_class: Type):
        if loader_class in self.loaders:
            del self.loaders[loader_class]
        else:
            raise KeyError(f"Loader class '{loader_class}' is not registered")

    @validate_params(
        splitter_class=dict(validator=lambda x: isinstance(x, type), message=CLASS_TYPE_CHECK_TIP))
    def unregister_splitter(self, splitter_class: Type):
        if splitter_class in self.splitters:
            del self.splitters[splitter_class]
        else:
            raise KeyError(f"Splitter class '{splitter_class}' is not registered")
