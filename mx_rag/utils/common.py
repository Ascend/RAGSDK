# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import re


def is_english(texts):
    eng = 0
    if not texts:
        return False
    for t in texts:
        if re.match(r"[a-zA-Z]{2,}", t.strip()):
            eng += 1
    if eng / len(texts) > 0.8:
        return True
    return False


def safe_get(data, keys, default=None):
    """
    安全地获取嵌套字典或列表中的值。

    :param data: 字典或列表数据
    :param keys: 键或索引列表，表示嵌套层级
    :param default: 如果键或索引不存在，返回的默认值
    :return: 对应键或索引的值或默认值
    """
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        elif isinstance(data, list) and isinstance(key, int) and 0 <= key < len(data):
            data = data[key]
        else:
            return default
    return data
