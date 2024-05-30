# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os


class SizeOverLimitException(Exception):
    pass


class PathNotFileException(Exception):
    pass


class PathNotDirException(Exception):
    pass


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
