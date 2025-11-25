#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import unittest
from pathlib import Path
from typing import List, Tuple, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter

from mx_rag.document.loader import DocxLoader, ExcelLoader, PdfLoader, MarkdownLoader
from mx_rag.document.splitter import MarkdownTextSplitter

DOC_PARSER_MAP = {
    ".docx": (DocxLoader, RecursiveCharacterTextSplitter),
    ".xlsx": (ExcelLoader, RecursiveCharacterTextSplitter),
    ".xls": (ExcelLoader, RecursiveCharacterTextSplitter),
    ".csv": (ExcelLoader, RecursiveCharacterTextSplitter),
    ".pdf": (PdfLoader, RecursiveCharacterTextSplitter),
    ".md": (MarkdownLoader, MarkdownTextSplitter),
}


class TestTokenParseDocumentFile(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.realpath(os.path.join(current_dir, "../../../data"))

    @staticmethod
    def token_parse_document_file(filepath, tokenizer, max_tokens, DOC_PARSER_MAP) -> Tuple[
        List[str], List[Dict[str, str]]]:
        file = Path(filepath)
        loader, splitter = DOC_PARSER_MAP.get(file.suffix, (None, None))
        if loader is None:
            raise ValueError(f"'{file.suffix}' is not support")
        metadatas = []
        texts = []
        for doc in loader(file.as_posix()).load():
            split_texts = ["test_txt"]
            metadatas.extend(doc.metadata for _ in split_texts)
            texts.extend(split_texts)
        return texts, metadatas

    def setUp(self):
        self.file_path = os.path.join(os.path.join(self.data_dir, "demo.docx"))

    def test_token_parse_document_file_unsupported_file_type(self):
        with self.assertRaises(ValueError):
            TestTokenParseDocumentFile.token_parse_document_file(
                os.path.join(os.path.join(self.data_dir, "Sample.img")),
                None, 100, DOC_PARSER_MAP)

    def test_token_parse_document_file_sample(self):
        tokenizer = None
        texts, metadatas = TestTokenParseDocumentFile.token_parse_document_file(self.file_path, tokenizer, 100,
                                                                                DOC_PARSER_MAP)
        self.assertEqual(metadatas, [{'source': 'demo.docx', 'type': 'text'}])
