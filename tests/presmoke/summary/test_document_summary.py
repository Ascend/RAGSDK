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

from langchain_text_splitters import RecursiveCharacterTextSplitter

from mx_rag.document.loader import DocxLoader
from mx_rag.llm import Text2TextLLM
from mx_rag.summary import Summary
from mx_rag.utils import ClientParam


class TestDocumentSummary(unittest.TestCase):
    def test_document_summary(self):
        test_file_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                       "../../data/test.docx"))
        client_param = ClientParam(use_http=True)
        llm = Text2TextLLM(base_url="http://127.0.0.1:1025/v1/chat/completions", model_name="Llama3-8B-Chinese-Chat",
                           client_param=client_param)
        loader = DocxLoader(test_file_path)
        docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=150))
        summary = Summary(llm=llm)
        # 调用summarize方法
        sub_summaries = summary.summarize([doc.page_content for doc in docs])
        # 调用merge_text_summarize方法
        res = summary.merge_text_summarize(sub_summaries)
        self.assertGreater(len(res), 0)
