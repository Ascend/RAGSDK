# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from abc import ABC, abstractmethod

import zipfile
from loguru import logger


class BaseLoader(ABC):
    MAX_SIZE = 100 * 1024 * 1024
    MAX_PAGE_NUM = 1000

    def __init__(self, file_path):
        self.file_path = file_path
        self.multi_size = 5

    @abstractmethod
    def load(self):
        pass

    def _is_zip_bomb(self):
        try:
            with zipfile.ZipFile(self.file_path, "r") as zip_ref:
                total_uncompressed_size = sum(zinfo.file_size for zinfo in zip_ref.infolist())
                if total_uncompressed_size > self.MAX_SIZE * self.multi_size:
                    logger.error(f"{self.file_path} is ZIP bomb: file is too large after decompression.")
                    return True
                else:
                    return False
        except Exception as e:
            logger.error(f"Error checking ZIP bomb: {e}")
            return True
