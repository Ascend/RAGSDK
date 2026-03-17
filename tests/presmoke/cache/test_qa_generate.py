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

from transformers import AutoTokenizer

from mx_rag.cache import QAGenerate, QAGenerationConfig
from mx_rag.llm import Text2TextLLM
from mx_rag.utils import ClientParam


class TestQAGenerate(unittest.TestCase):
    def test_generate_qa(self):
        llm = Text2TextLLM(base_url="http://127.0.0.1:8000/v1/chat/completions_qa_generate",
                           model_name="llama3-chinese-8b-chat",
                           client_param=ClientParam(use_http=True))
        # 使用模型的tokenizer, 传入模型存放路径
        tokenizer = AutoTokenizer.from_pretrained("/home/data/Llama3-8B-Chinese-Chat", local_files_only=True)
        # 可以调用MarkDownParser生成titles和contents
        titles = ["2024年高考语文作文题目"]
        contents = ['2024年高考语文作文试题\n新课标I卷\n阅读下面的材料，根据要求写作。（60分）\n'
                    '随着互联网的普及、人工智能的应用，越来越多的问题能很快得到答案。那么，我们的问题是否会越来越少？\n'
                    '以上材料引发了你怎样的联想和思考？请写一篇文章。'
                    '要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不得泄露个人信息；不少于800字。']
        config = QAGenerationConfig(titles, contents, tokenizer, llm, qas_num=1)
        qa_generate = QAGenerate(config)
        qas = qa_generate.generate_qa()
        self.assertEqual(qas, {'2024年高考语文作文题目是什么？': '新课标Ⅰ卷。'})

if __name__ == '__main__':
    unittest.main()