#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
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
from unittest.mock import MagicMock, patch

from langchain.text_splitter import RecursiveCharacterTextSplitter

from mx_rag.document.loader import MarkdownLoader
from mx_rag.llm import Img2TextLLM
from mx_rag.utils.file_check import FileCheckError


class MarkdownLoaderTestCase(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.realpath(os.path.join(current_dir, "../../../data"))

    def test_load_basic_markdown(self):
        """Test loading a basic markdown file"""
        loader = MarkdownLoader(os.path.join(self.data_dir, "example.md"))
        docs = loader.load()
        self.assertEqual(1, len(docs))
        self.assertIn("这是无序列表", docs[0].page_content)

    def test_load_with_images(self):
        """Test loading a markdown file with images"""
        mock_vlm = MagicMock(spec=Img2TextLLM)
        mock_vlm.chat = MagicMock(return_value="Image description")

        loader = MarkdownLoader(
            os.path.join(self.data_dir, "example.md"),
            vlm=mock_vlm
        )
        docs = loader.load()
        self.assertEqual(1, len(docs))
        self.assertIn("Image description", docs[0].page_content)

    def test_process_images_separately(self):
        """Test processing images separately"""
        mock_vlm = MagicMock(spec=Img2TextLLM)
        mock_vlm.chat = MagicMock(return_value="Image description")

        loader = MarkdownLoader(
            os.path.join(self.data_dir, "example.md"),
            vlm=mock_vlm,
            process_images_separately=True
        )
        docs = list(loader.lazy_load())  # Convert to list for counting
        self.assertGreater(len(docs), 1)  # Should include text content and separate image documents


    def test_load_and_split(self):
        """Test loading and splitting markdown content"""
        loader = MarkdownLoader(os.path.join(self.data_dir, "example.md"))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = loader.load_and_split(text_splitter)
        self.assertGreater(len(chunks), 1)

    def test_with_tables(self):
        """Test a markdown file with tables"""
        loader = MarkdownLoader(os.path.join(self.data_dir, "example.md"))
        docs = loader.load()
        self.assertEqual(1, len(docs))
        self.assertIn("第二格表头:内容单元格第二列第二格", docs[0].page_content)

    def test_invalid_file_type(self):
        """Test handling of invalid file types"""
        with self.assertRaises(TypeError):
            loader = MarkdownLoader(os.path.join(self.data_dir, "test.txt"))
            loader.load()

    def test_large_file(self):
        """Test handling of large files"""
        # Create a temporary large file
        test_file = os.path.join(self.data_dir, "large_temp.md")
        with open(test_file, "w") as f:
            f.write("# Large file test\n" + "Test content " * 100 * 1024 * 1024)

        with self.assertRaises(FileCheckError):
            loader = MarkdownLoader(test_file)
            loader.load()
        os.remove(test_file)


if __name__ == '__main__':
    unittest.main()