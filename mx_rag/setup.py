# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize([
    'document/loader/docx_loader.py',
    'document/loader/data_clean.py'
]))
