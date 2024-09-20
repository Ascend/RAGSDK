# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import unittest

from mx_rag.utils import ClientParam
from mx_rag.utils.url import UrlError, _is_url_valid, RequestUtils


class TestURL(unittest.TestCase):
    def test_is_url_valid(self):
        self.assertFalse(_is_url_valid("https://www.google.com", True))
        self.assertTrue(_is_url_valid("https://www.google.com", False))
        self.assertTrue(_is_url_valid("http://www.google.com", True))
        with self.assertRaises(UrlError):
            self.assertFalse(_is_url_valid("http://www.google.com", False))
        self.assertFalse(_is_url_valid("not a url", True))
