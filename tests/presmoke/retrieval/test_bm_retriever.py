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

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mx_rag.chain import SingleText2TextChain
from mx_rag.llm import Text2TextLLM
from mx_rag.retrievers.bm_retriever import BMRetriever
from mx_rag.utils import ClientParam


class TestBMRetriever(unittest.TestCase):
    def test_bm_retriever(self):
        test_file_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                       "../../data/test.txt"))
        loader = TextLoader(test_file_path)
        docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=150))
        client_param = ClientParam(use_http=True)
        llm = Text2TextLLM(base_url="http://127.0.0.1:1025/v1/chat/completions",
                           model_name="Llama3-8B-Chinese-Chat", client_param=client_param)
        bm_retriever = BMRetriever(docs=docs, llm=llm, k=10)
        text2text_chain = SingleText2TextChain(llm=llm, retriever=bm_retriever)
        res = text2text_chain.query("小明去哪里?")
        self.assertGreater(len(res.get("result")), 0)
        self.assertGreater(len(res.get("source_documents")), 0)


if __name__ == '__main__':
    unittest.main()
