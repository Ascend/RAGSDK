#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import unittest
from unittest.mock import patch

from mx_rag.document.loader.load_excel import ExcelLoader


class TestExcelLoader(unittest.TestCase):
    def test_class_init_case(self):
        loader = ExcelLoader("./document/loader/files/test.xlsx")
        self.assertIsInstance(loader, ExcelLoader)
