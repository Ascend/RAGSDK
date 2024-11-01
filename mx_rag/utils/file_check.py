# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import shutil
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
        FileCheck.check_path_is_exist_and_valid(self.file_path)

        if not os.path.isfile(self.file_path):
            raise PathNotFileException(f"PathNotFileException: '{self.file_path}' is not file")

        FileCheck.check_file_size(self.file_path, self.max_size)
        FileCheck.check_file_owner(self.file_path)


class FileCheck:
    MAX_PATH_LENGTH = 1024
    DEFAULT_MAX_FILE_NAME_LEN = 255
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
        if len(file_path) > FileCheck.MAX_PATH_LENGTH:
            raise FileCheckError(f"Input path '{file_path[:FileCheck.MAX_PATH_LENGTH]}...' length over limit")
        file_size = os.path.getsize(file_path)
        if file_size > max_file_size:
            raise FileCheckError(f"FileSizeLimit: '{file_path}' size over Limit: {max_file_size}")

    @staticmethod
    def check_input_path_valid(path: str, check_real_path: bool = True, check_blacklist: bool = False):
        if not path or not isinstance(path, str):
            raise FileCheckError(f"Input path '{path}' is not valid str")

        if len(path) > FileCheck.MAX_PATH_LENGTH:
            raise FileCheckError(f"Input path '{path[:FileCheck.MAX_PATH_LENGTH]}'... length over limit")

        if ".." in path:
            raise FileCheckError(f"there are illegal characters in path '{path}'")

        if check_real_path and Path(path).resolve() != Path(path).absolute():
            raise FileCheckError(f"Input path '{path}' is not valid")
        path_obj = Path(path)
        if check_blacklist:
            for black_path in FileCheck.BLACKLIST_PATH:
                if path_obj.resolve().is_relative_to(black_path):
                    raise FileCheckError(f"Input path '{path}' is in blacklist")

    @staticmethod
    def check_path_is_exist_and_valid(path: str, check_real_path: bool = True):
        if not isinstance(path, str):
            raise FileCheckError(f"Input path '{path}' is not valid str")
        
        if len(path) > FileCheck.MAX_PATH_LENGTH:
            raise FileCheckError(f"Input path '{path[:FileCheck.MAX_PATH_LENGTH]}...' length over limit")

        if not os.path.exists(path):
            raise FileCheckError(f"path '{path}' is not exists")

        FileCheck.check_input_path_valid(path, check_real_path)

    @staticmethod
    def dir_check(file_path: str):
        if len(file_path) > FileCheck.MAX_PATH_LENGTH:
            raise FileCheckError(f"Input path '{file_path[:FileCheck.MAX_PATH_LENGTH]}...' length over limit")

        if not file_path.startswith("/"):
            raise FileCheckError(f"dir '{file_path}' must be an absolute path")

        if not os.path.isdir(file_path):
            raise PathNotDirException(f"PathNotDirException: ['{file_path}'] is not a valid dir")

        FileCheck.check_input_path_valid(file_path, True)

    @staticmethod
    def check_files_num_in_directory(directory_path: str, suffix: str, limit: int):
        if len(directory_path) > FileCheck.MAX_PATH_LENGTH:
            raise FileCheckError(f"Input path '{directory_path[:FileCheck.MAX_PATH_LENGTH]}...' length over limit")
        count = sum(1 for file in Path(directory_path).glob("*") if not suffix or file.suffix == suffix)
        if count > limit:
            raise FileCheckError(f"The number of '{suffix}' files in '{directory_path}' exceed {limit}")

    @staticmethod
    def check_file_owner(file_path: str):
        if len(file_path) > FileCheck.MAX_PATH_LENGTH:
            raise FileCheckError(f"Input path '{file_path[:FileCheck.MAX_PATH_LENGTH]}...' length over limit")
        current_user_uid = os.getuid()

        def check_owner(path: str, path_type: str):
            """辅助函数，用于检查一个文件或目录的属主。"""
            try:
                stat_info = os.stat(path)
                owner_uid = stat_info.st_uid
                if owner_uid != current_user_uid:
                    raise FileCheckError(f"The owner of the {path_type} '{path}' is different from the current user")
            except FileNotFoundError as fnf_error:
                raise FileCheckError(f"The {path_type} '{path}' does not exist") from fnf_error
            except PermissionError as pe_error:
                raise FileCheckError(f"Permission denied when accessing the {path_type} '{path}'") from pe_error

        # 检查文件的属主
        check_owner(file_path, "file")

        # 获取文件所在的目录
        dir_path = os.path.dirname(os.path.abspath(file_path))
        # 检查目录的属主
        check_owner(dir_path, "directory")
    
    @staticmethod
    def check_filename_valid(file_path: str, max_length: int = 0):
        max_length = FileCheck.DEFAULT_MAX_FILE_NAME_LEN if max_length <= 0 else max_length
        file_name = os.path.basename(file_path)
        if len(file_name) > max_length:
            raise FileCheckError(f"the file name length of {file_name[:max_length]}... is over limit {max_length}")


def check_disk_free_space(path, volume):
    _, _, free = shutil.disk_usage(path)
    return free < volume