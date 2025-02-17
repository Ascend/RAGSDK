# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import unittest

from mx_rag.tools.finetune.dataprocess.reciprocal_rank_fusion import reciprocal_rank_fusion


class TestRRF(unittest.TestCase):

    def test_run_success(self):
        a = ["1", "2", "3"]
        b = ["2", "3", "4"]

        c = reciprocal_rank_fusion([a, b])
        self.assertEqual(len(c), 4)


if __name__ == '__main__':
    unittest.main()
