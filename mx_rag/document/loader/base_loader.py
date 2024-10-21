# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
from abc import ABC, abstractmethod
import zipfile
import psutil

from loguru import logger

from mx_rag.utils import file_check
from mx_rag.utils.common import validate_params, STR_TYPE_CHECK_TIP_1024


class BaseLoader(ABC):
    MAX_SIZE = 100 * 1024 * 1024
    MAX_PAGE_NUM = 1000
    MAX_WORD_NUM = 500000
    MAX_FILE_CNT = 1024

    @validate_params(
        file_path=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024, message=STR_TYPE_CHECK_TIP_1024),
    )
    def __init__(self, file_path):
        self.file_path = file_path
        self.multi_size = 5
        file_check.SecFileCheck(self.file_path, self.MAX_SIZE).check()

    def _is_zip_bomb(self):
        try:
            with zipfile.ZipFile(self.file_path, "r") as zip_ref:
                # 检查点1：检查文件个数，文件个数大于预期值时上报异常退出
                file_count = len(zip_ref.infolist())
                if file_count >= self.MAX_FILE_CNT * self.multi_size:
                    logger.error(f'zip file ({self.file_path}) contains {file_count} files, exceed '
                                 f'the limit of {self.MAX_FILE_CNT * self.multi_size}')
                    return True
                # 检查点2：检查第一层解压文件总大小，总大小超过设定的上限值
                # total_uncompressed_size = sum(zinfo.file_size for zinfo in zip_ref.infolist())
                total_uncompressed_size = 0
                for zinfo in zip_ref.infolist():
                    total_uncompressed_size += zinfo.file_size
                    if total_uncompressed_size > self.MAX_SIZE * self.multi_size:
                        logger.error(f"zip file '{self.file_path}' uncompressed size is {total_uncompressed_size} bytes"
                                     f"exceeds the limit of {self.MAX_SIZE * self.multi_size} bytes, Potential ZIP bomb")
                        return True
                # 检查点3：检查第一层解压文件总大小，磁盘剩余空间-文件总大小<200M
                remain_size = psutil.disk_usage(os.getcwd()).free
                if remain_size - total_uncompressed_size < self.MAX_SIZE * 2:
                    logger.error(f'zip file ({self.file_path}) uncompressed size is {total_uncompressed_size} bytes'
                                 f' only {remain_size} bytes of disk space available')
                    return True

                return False
        except zipfile.BadZipfile as e:
            logger.error(f"The provided path '{self.file_path}' is not a valid ZIP file or is corrupted: {e}")
            return True
        except Exception as e:
            logger.error(f"Unexpected error occurred while checking ZIP bomb: {e}")
            return True
