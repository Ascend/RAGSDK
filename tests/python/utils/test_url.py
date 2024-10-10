# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import unittest

from mx_rag.utils.url import is_url_valid


class TestURL(unittest.TestCase):
    def test_is_url_valid(self):
        self.assertFalse(is_url_valid("https://www.google.com", True))
        self.assertTrue(is_url_valid("https://www.google.com", False))
        self.assertTrue(is_url_valid("http://www.google.com", True))
        self.assertFalse(is_url_valid("http://www.google.com", False))
        self.assertFalse(is_url_valid("not a url", True))
