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
    def upload_files(self, *args, **kwargs):
        pass

    @abstractmethod
    def upload_dir(self, *args, **kwargs):
        pass

    @abstractmethod
    def delete_files(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_all_documents(self, *args, **kwargs):
        pass


class KnowledgeError(Exception):
    def __init__(self, err_msg: str):
        self.err_msg = err_msg
        info: inspect.FrameInfo = inspect.stack()[1]
        msg = f"{info.function}({Path(info.filename).name}:{info.lineno})-{err_msg}"
        super().__init__(msg)
