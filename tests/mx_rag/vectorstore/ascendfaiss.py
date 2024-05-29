# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.


class AscendIndexFlatL2:
    def __init__(self, *args, **kwargs):
        pass

    def add_with_ids(self, *args, **kwargs):
        pass

    def search(self, *args, **kwarg):
        return [[[0.1]]], [[0]]

    def remove_ids(self, *args, **kwarg):
        pass


class AscendIndexFlatConfig:
    def __init__(self, *args, **kwargs):
        pass


class IntVector:
    def __init__(self, *args, **kwargs):
        pass


def index_cpu_to_ascend(*args, **kwargs):
    return AscendIndexFlatL2()


def index_ascend_to_cpu(*args, **kwargs):
    return ""
