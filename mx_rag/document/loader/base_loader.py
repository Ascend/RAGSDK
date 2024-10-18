# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from abc import ABC, abstractmethod
import psutil

import zipfile
from loguru import logger
from mx_rag.utils.common import validate_params, STR_TYPE_CHECK_TIP

class BaseLoader(ABC):
    MAX_SIZE = 100 * 1024 * 1024
    MAX_PAGE_NUM = 1000
    MAX_FILE_CNT = 1024*5

    @validate_params(
        file_path=dict(validator=lambda x: isinstance(x, str), message=STR_TYPE_CHECK_TIP),
    )
    def __init__(self, file_path):
        self.file_path = file_path
        self.multi_size = 5


    def _is_zip_bomb(self):
        try:
            with zipfile.ZipFile(self.file_path, "r") as zip_ref:
                # 检查点1：检查文件个数，文件个数大于预期值时上报异常退出
                file_count = len(zip_ref.infolist())
                if file_count >= self.MAX_FILE_CNT:
                    logger.error(f'zipfile({self.file_path}) contains {file_count} files exceed max file count ')
                    return True
                # 检查点2：检查第一层解压文件总大小，总大小超过设定的上限值
                total_uncompressed_size = sum(zinfo.file_size for zinfo in zip_ref.infolist())
                if total_uncompressed_size > self.MAX_SIZE * self.multi_size:
                    logger.error(f"'{self.file_path}' is ZIP bomb: file is too large after decompression.")
                    return True
                # 检查点3：检查第一层解压文件总大小，磁盘剩余空间-文件总大<200M
                remain_size = psutil.disk_usage('/').free
                if remain_size - total_uncompressed_size < self.MAX_SIZE:
                    logger.error(f'zipfile({self.file_path}) size is ({total_uncompressed_size})'
                                 f' only {remain_size} disk space available')
                    return True

                return False
        except zipfile.BadZipfile as e:
            logger.error(f"The provided path '{self.file_path}' is not a valid ZIP file or is corrupted: {e}")
            return True
        except Exception as e:
            logger.error(f"Unexpected error occurred while checking ZIP bomb: {e}")
            return True
