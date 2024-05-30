#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import unittest

from langchain.text_splitter import CharacterTextSplitter

from mx_rag.document.loader.pdf_loader import PdfLoader
from mx_rag.document.spliter.char_text_splitter import CharTextSplitter


class TestPdfSplit(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    def test_pdf_split(self):
        loader = PdfLoader(os.path.join(self.current_dir, "../../../data/test.pdf"))
        pdf_doc = loader.load()

        langchain_char_text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=100, separator="\n")
        expect_split_text = langchain_char_text_splitter.split_text(pdf_doc[0].page_content)

        mxrag_char_text_splitter = CharTextSplitter(chunk_size=512, chunk_overlap=100, separator="\n")
        result_split_text = mxrag_char_text_splitter.split_text(pdf_doc[0].page_content)

        self.assertEqual(expect_split_text, result_split_text)
