# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import inspect
from abc import ABC, abstractmethod
from typing import List

from mx_rag.utils.common import validate_params, validata_list_str, MAX_PATH_WHITE


class KnowledgeBase(ABC):
    @validate_params(
        white_paths=dict(validator=lambda x: validata_list_str(x, [1, MAX_PATH_WHITE], [1, MAX_PATH_WHITE]),
                         message="param must meets: Type is List[str], "
                                 "list length range [1, 1024], str length range [1, 1024]"))
    def __init__(self, white_paths: List[str]):
        self.white_paths = white_paths

    @abstractmethod
    def add_file(self, doc_name, texts, embed_func, metadatas):
        pass

    @abstractmethod
    def check_document_exist(self, doc_name):
        pass

    @abstractmethod
    def delete_file(self, doc_name):
        pass

    @abstractmethod
    def get_all_documents(self):
        pass


class KnowledgeError(Exception):
    pass
