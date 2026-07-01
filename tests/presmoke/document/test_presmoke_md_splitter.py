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

from mx_rag.document.splitter.md_splitter import MarkdownTextSplitter


class TestMarkdownSplitter(unittest.TestCase):
    def test_code_block_not_split(self):
        code_content = "```python\nx = 1\ny = 2\n```"
        text = "# Header\n\n" + code_content
        splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=10)
        chunks = splitter.split_text(text)
        code_found_whole = any(code_content in chunk for chunk in chunks)
        self.assertTrue(code_found_whole)

    def test_markdown_table_not_split(self):
        table_content = "| col1 | col2 |\n| --- | --- |\n| a | b |"
        text = "# Header\n\n" + table_content
        splitter = MarkdownTextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split_text(text)
        table_found = any("| col1 |" in chunk and "| --- |" in chunk for chunk in chunks)
        self.assertTrue(table_found)

    def test_html_table_not_split(self):
        table_content = "<table><tr><td>cell</td></tr></table>"
        text = "# Header\n\n" + table_content
        splitter = MarkdownTextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split_text(text)
        table_found = any("<table>" in chunk and "</table>" in chunk for chunk in chunks)
        self.assertTrue(table_found)

    def test_empty_input(self):
        splitter = MarkdownTextSplitter()
        chunks = splitter.split_text("")
        self.assertEqual(len(chunks), 1)

    def test_small_text_no_split(self):
        text = "short text"
        splitter = MarkdownTextSplitter(chunk_size=500)
        chunks = splitter.split_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "short text")


if __name__ == '__main__':
    unittest.main()
