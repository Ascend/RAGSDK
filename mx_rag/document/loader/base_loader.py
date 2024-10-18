# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from abc import ABC, abstractmethod
import psutil

import zipfile
from loguru import logger


class BaseLoader(ABC):
    MAX_SIZE = 100 * 1024 * 1024
    MAX_PAGE_NUM = 1000
    MAX_WORD_NUM = 500000
    MAX_FILE_CNT = 1000

    def __init__(self, file_path):
        self.file_path = file_path
        self.multi_size = 5

    @abstractmethod
    def load(self):
        pass

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
                # ？检查点3：检查第一层解压文件总大小，总大小超过磁盘剩余空间？200M
                if total_uncompressed_size > psutil.disk_usage('/').free:
                    logger.error(f'zipfile({self.file_path}) size ({total_uncompressed_size})'
                                 f' exceed remain target disk space')
                    return True

                return False
        except Exception as e:
            logger.error(f"Error checking ZIP bomb: {e}")
            return True
