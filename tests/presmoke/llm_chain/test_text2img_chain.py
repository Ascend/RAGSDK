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

import unittest

from mx_rag.chain import Text2ImgChain
from mx_rag.llm import Text2ImgMultiModel
from mx_rag.utils import ClientParam


class TestText2ImgChain(unittest.TestCase):
    def test_text2img_chain(self):
        multi_model = Text2ImgMultiModel(model_name="sd", url="http://127.0.0.1:8000/text2img",
                                         client_param=ClientParam(use_http=True))
        text2img_chain = Text2ImgChain(multi_model=multi_model)
        llm_data = text2img_chain.query("dog wearing black glasses", output_format="jpg")
        self.assertEqual(str(llm_data.get("prompt")), "dog wearing black glasses")


if __name__ == '__main__':
    unittest.main()
