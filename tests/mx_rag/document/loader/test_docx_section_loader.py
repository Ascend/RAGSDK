# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import unittest

from langchain.text_splitter import RecursiveCharacterTextSplitter

from mx_rag.document.loader.docx_section_loader import DocxLoaderByHead


class DocxSectionLoaderTestCase(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    def test_load(self):
        loader = DocxLoaderByHead(os.path.join(self.current_dir, "../../../data/demo.docx"))
        res = loader.load()
        self.assertEqual(1, len(res))

    def test_load_and_split(self):
        loader = DocxLoaderByHead(os.path.join(self.current_dir, "../../../data/demo.docx"))
        res = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100))
        self.assertEqual(2, len(res))

    def test_title(self):
        loader = DocxLoaderByHead(os.path.join(self.current_dir, "../../../data/title.docx"))
        res = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100))
        self.assertEqual(1, len(res))

    def test_link(self):
        loader = DocxLoaderByHead(os.path.join(self.current_dir, "../../../data/link.docx"))
        res = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100))
        self.assertEqual(7, len(res))


if __name__ == '__main__':
    unittest.main()
