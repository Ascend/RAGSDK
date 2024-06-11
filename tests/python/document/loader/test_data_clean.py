# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import unittest

from mx_rag.document.loader.data_clean import (
    is_alpha_or_digit_or_punc,
    remove_special_char,
    remove_duplicate_punctuation,
    process_sentence
)


class DataCleanTestCase(unittest.TestCase):
    def test_is_alpha_or_digit_or_punc(self):
        for c in "aZ1用！":
            ret = is_alpha_or_digit_or_punc(c)
            self.assertTrue(ret)

    def test_remove_special_char(self):
        ret = remove_special_char('a')
        self.assertEqual(ret, "a")

    def test_remove_duplicate_punctuation(self):
        ret = remove_duplicate_punctuation('a，，')
        self.assertEqual(ret, "a，")

    def test_process_sentence(self):
        ret = process_sentence('a，，')
        self.assertEqual(ret, "a，")


if __name__ == '__main__':
    unittest.main()
