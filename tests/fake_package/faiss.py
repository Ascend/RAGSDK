# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import numpy as np


def write_index(*args, **kwargs):
    return ""


def read_index(*args, **kwargs):
    return ""


class IndexFlatIP:
    def __init__(self, embed_len: int):
        self.embed_len = embed_len

    def add(self, embedding: list):
        pass

    def search(self, batch_embedding: list, k: int):
        return np.array([i for i in range(len(batch_embedding))]), \
               np.array([[i for i in range(k)]] * len(batch_embedding))
