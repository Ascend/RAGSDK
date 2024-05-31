# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import inspect
import os
import re
from pathlib import Path


class SizeOverLimitException(Exception):
    pass


class PathNotFileException(Exception):
    pass


class PathNotDirException(Exception):
    pass


class FileCheckError(Exception):
    def __init__(self, err_msg: str):
        self.err_msg = err_msg
        info: inspect.FrameInfo = inspect.stack()[1]
        msg = f"{info.function}({Path(info.filename).name}:{info.lineno})-{err_msg}"
        super().__init__(msg)



class SecFileCheck:
    def __init__(self, file_path, max_size_mb):
        self.file_path = file_path
        self.max_size_mb = max_size_mb

    def check(self):
        if not os.path.isfile(self.file_path):
            raise PathNotFileException(f"PathNotFileException: {self.file_path} is not file")
        self._check_size()
        return None

    def _check_size(self):
        file_size = os.path.getsize(self.file_path)
        if file_size > self.max_size_mb * 1024 * 1024:
            raise SizeOverLimitException(
                f"SizeOverLimitException: {self.file_path} size over Limit: {self.max_size_mb}")


def excel_file_check(file_path, size):
    file_check = SecFileCheck(file_path, size)
    return file_check.check()


def dir_check(file_path):
    if not os.path.isdir(file_path):
        raise PathNotDirException(f"PathNotDirException: [{file_path}] is not a valid dir")


class FileCheck:
    MAX_PATH_LENGTH = 1024

    @staticmethod
    def check_input_path_valid(path: str, check_real_path: bool = True):
        if not path or not isinstance(path, str):
            raise FileCheckError("Input path is not valid str")

        if len(path) > FileCheck.MAX_PATH_LENGTH:
            raise FileCheckError("Input path length over limit")

        FileCheck._check_normal_file_path(path)

        if check_real_path and os.path.islink(path):
            raise FileCheckError("Input path is symbol link")

    @staticmethod
    def check_path_is_exist_and_valid(path: str, check_real_path: bool = True):
        if not isinstance(path, str) or not os.path.exists(path):
            raise FileCheckError("path is not exists")

        FileCheck.check_input_path_valid(path, check_real_path)

    @staticmethod
    def _check_normal_file_path(path):
        pattern_name = re.compile(r"[^0-9a-zA-Z_./-]")
        match_name = pattern_name.findall(path)
        if match_name or ".." in path:
            raise FileCheckError("there are illegal characters in path")
