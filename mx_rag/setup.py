# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import glob
import os
from distutils.core import setup

from Cython.Build import cythonize


def get_python_files():
    src_dir = os.path.dirname(os.path.realpath(__file__))
    file_list = []
    for name in glob.glob(f"{src_dir}/**/*.py", recursive=True):
        file = name[len(src_dir) + 1:]
        file_list.append(file)
    return file_list


if __name__ == "__main__":
    setup(ext_modules=cythonize(get_python_files()))
