# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
    pass


class SecFileCheck:
    def __init__(self, file_path, max_size):
        self.file_path = file_path
        self.max_size = max_size

    def check(self):
        if not os.path.isfile(self.file_path):
            raise PathNotFileException(f"PathNotFileException: {self.file_path} is not file")
        self._check_size()
        return None

    def _check_size(self):
        file_size = os.path.getsize(self.file_path)
        if file_size > self.max_size:
            raise SizeOverLimitException(
                f"SizeOverLimitException: {self.file_path} size over Limit: {self.max_size}")


def excel_file_check(file_path, size):
    file_check = SecFileCheck(file_path, size)
    return file_check.check()


class FileCheck:
    MAX_PATH_LENGTH = 1024
    BLACKLIST_PATH = [
        "/etc/",
        "/usr/bin/",
        "/usr/lib/",
        "/usr/lib64/",
        "/sys/",
        "/dev/",
        "/sbin",
    ]

    @staticmethod
    def check_file_size(file_path: str, max_file_size: int):
        file_size = os.path.getsize(file_path)
        if file_size > max_file_size:
            raise FileCheckError(f"FileSizeLimit: {file_path} size over Limit: {max_file_size}")

    @staticmethod
    def check_input_path_valid(path: str, check_real_path: bool = True, check_blacklist: bool = False):
        if not path or not isinstance(path, str):
            raise FileCheckError(f"Input path {path} is not valid str")

        if len(path) > FileCheck.MAX_PATH_LENGTH:
            raise FileCheckError(f"Input path {path} length over limit")

        FileCheck._check_normal_file_path(path)

        if check_real_path and os.path.islink(path):
            raise FileCheckError(f"Input path {path} is symbol link")
        path_obj = Path(path)
        if check_blacklist:
            for black_path in FileCheck.BLACKLIST_PATH:
                if path_obj.resolve().is_relative_to(black_path):
                    raise FileCheckError(f"Input path {path} is in blacklist")

    @staticmethod
    def check_path_is_exist_and_valid(path: str, check_real_path: bool = True):
        if not isinstance(path, str) or not os.path.exists(path):
            raise FileCheckError("path is not exists")

        FileCheck.check_input_path_valid(path, check_real_path)

    @staticmethod
    def dir_check(file_path: str):
        if not file_path.startswith("/"):
            raise FileCheckError("dir must be an absolute path")

        if not os.path.isdir(file_path):
            raise PathNotDirException(f"PathNotDirException: [{file_path}] is not a valid dir")

        FileCheck.check_input_path_valid(file_path, True)

    @staticmethod
    def check_files_num_in_directory(directory_path: str, suffix: str, limit: int):
        files = os.listdir(directory_path)
        filtered_files = [file for file in files if file.endswith(suffix)]
        if len(filtered_files) > limit:
            raise FileCheckError(f"The number of {suffix} files in {directory_path} exceed {limit}")

    @staticmethod
    def _check_normal_file_path(path):
        pattern_name = re.compile(r"[^0-9a-zA-Z_./-]")
        match_name = pattern_name.findall(path)
        if match_name or ".." in path:
            raise FileCheckError("there are illegal characters in path")
