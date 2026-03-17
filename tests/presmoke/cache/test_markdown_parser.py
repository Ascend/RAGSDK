#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

RAGSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import os
import unittest

from mx_rag.cache import MarkDownParser


class TestMarkDownParser(unittest.TestCase):
    def test_parse(self):
        test_dir_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data"))
        parser = MarkDownParser(test_dir_path)
        titles, contents = parser.parse()
        self.assertGreater(titles.index("test.md"), -1)


if __name__ == '__main__':
    unittest.main()