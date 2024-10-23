# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
import os
import re
import stat

from loguru import logger

from .file_check import FileCheck, SecFileCheck

R_FLAGS = os.O_RDONLY
MODES = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH


def read_jsonl_from_file(file: str,
                         file_size: int = 100 * 1024 * 1024,
                         line_limit: int = 10000,
                         length_limit: int = 1024):

    SecFileCheck(file, file_size).check()

    datas = []
    line_cnt = 0
    try:
        datas = _read_jsonl_file(file, length_limit, line_cnt, line_limit)
    except json.JSONDecodeError as json_decode_e:
        logger.error(f"read data from file failed, find JSONDecodeError: {json_decode_e}")
    except Exception as e:
        logger.error(f"read data from file failed, find Exception: {e}")
    return datas


def _read_jsonl_file(file, length_limit, line_cnt, line_limit):
    datas = []
    with os.fdopen(os.open(file, R_FLAGS, MODES), 'r') as f:
        while line_cnt < line_limit:
            line_cnt += 1

            line = f.readline()
            if len(line) > length_limit:
                logger.warning(f"data logger than {length_limit}, is {len(line)}")
                continue
            # 最后一行为空字符串
            if line == "" or line.isspace():
                break
            data = json.loads(line)
            datas.append(data)

        if line_cnt >= line_limit:
            logger.warning(f"read data more than {line_limit} lines")
    return datas