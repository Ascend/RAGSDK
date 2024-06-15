# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Callable


class KnowledgeBase(ABC):

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
    def __init__(self, err_msg: str):
        self.err_msg = err_msg
        info: inspect.FrameInfo = inspect.stack()[1]
        msg = f"{info.function}({Path(info.filename).name}:{info.lineno})-{err_msg}"
        super().__init__(msg)
