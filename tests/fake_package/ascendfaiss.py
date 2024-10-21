# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import numpy as np


class AscendIndexFlat:
    def __init__(self, *args, **kwargs):
        self.ntotal = 0
        pass

    def add_with_ids(self, *args, **kwargs):
        pass

    def search(self, *args, **kwarg):
        return np.array([[0.1]]), np.array([[0]])

    def remove_ids(self, *args, **kwarg):
        pass


class AscendIndexFlatConfig:
    def __init__(self, *args, **kwargs):
        pass


class IntVector:
    def __init__(self, *args, **kwargs):
        pass

    def push_back(self, *args, **kwargs):
        pass


def index_cpu_to_ascend(*args, **kwargs):
    return AscendIndexFlat()


def index_ascend_to_cpu(*args, **kwargs):
    return ""
