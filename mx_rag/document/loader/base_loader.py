# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from abc import ABC, abstractmethod

import zipfile
from loguru import logger

from mx_rag.utils import file_check
from mx_rag.utils.common import validate_params, STR_TYPE_CHECK_TIP_1024


class BaseLoader(ABC):
    MAX_SIZE = 100 * 1024 * 1024
    MAX_PAGE_NUM = 1000
    MAX_WORD_NUM = 500000

    @validate_params(
        file_path=dict(validator=lambda x: isinstance(x, str) and len(x) < 1024, message=STR_TYPE_CHECK_TIP_1024),
    )
    def __init__(self, file_path):
        self.file_path = file_path
        self.multi_size = 5
        file_check.SecFileCheck(self.file_path, self.MAX_SIZE).check()


    def _is_zip_bomb(self):
        try:
            with zipfile.ZipFile(self.file_path, "r") as zip_ref:
                total_uncompressed_size = sum(zinfo.file_size for zinfo in zip_ref.infolist())
                if total_uncompressed_size > self.MAX_SIZE * self.multi_size:
                    logger.error(f"'{self.file_path}' is ZIP bomb: file is too large after decompression.")
                    return True
                else:
                    return False
        except zipfile.BadZipfile as e:
            logger.error(f"The provided path '{self.file_path}' is not a valid ZIP file or is corrupted: {e}")
            return True
        except Exception as e:
            logger.error(f"Unexpected error occurred while checking ZIP bomb: {e}")
            return True
