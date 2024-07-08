# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import unittest

from docx import Document

from mx_rag.document.loader import DocxLoader
from mx_rag.document.splitter import CharTextSplitter


class DocxLoaderTestCase(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    def test_load(self):
        loader = DocxLoader(os.path.join(self.current_dir, "../../../data/demo.docx"))
        d = loader.load()
        self.assertEqual(1, len(d))

    def test_load_with_image(self):
        loader = DocxLoader(os.path.join(self.current_dir, "../../../data/demo.docx"), image_inline=True)
        d = loader.load()
        self.assertEqual(1, len(d))

    def test_load_and_split(self):
        loader = DocxLoader(os.path.join(self.current_dir, "../../../data/demo.docx"))
        res = loader.load_and_split(CharTextSplitter(chunk_size=512, chunk_overlap=100))
        self.assertEqual(2, len(res))


    def test_split(self):
        loader = DocxLoader(os.path.join(self.current_dir, "../../../data/demo.docx"))
        d = loader.load()
        splitter = CharTextSplitter(chunk_size=10, chunk_overlap=2, separator="\n\n")
        res = splitter.split_text(d[0].page_content)
        self.assertEqual(7, len(res))

    def test_title(self):
        loader = DocxLoader(os.path.join(self.current_dir, "../../../data/title.docx"))
        res = loader.load_and_split(CharTextSplitter(chunk_size=512, chunk_overlap=100))
        self.assertEqual(1, len(res))

    def test_link(self):
        loader = DocxLoader(os.path.join(self.current_dir, "../../../data/link.docx"))
        res = loader.load_and_split(CharTextSplitter(chunk_size=512, chunk_overlap=100))
        self.assertEqual(7, len(res))

    def test_page_number(self):
        document = Document()
        document.add_heading('Document Title', 0)

        idx = 0
        while idx <= 1000:
            idx += 1
            document.add_paragraph('A plain paragraph having some ')

        test_file = os.path.join(self.current_dir, "../../../data/page_number_test.docx")
        document.save(test_file)
        loader = DocxLoader(test_file)
        res = loader.load_and_split(CharTextSplitter(chunk_size=512, chunk_overlap=100))
        self.assertEqual(0, len(res))


if __name__ == '__main__':
    unittest.main()
