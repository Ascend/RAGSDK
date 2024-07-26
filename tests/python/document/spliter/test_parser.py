import os
import unittest
from pathlib import Path
from typing import List, Tuple, Dict
from unittest.mock import MagicMock, Mock

from mx_rag.document.loader import DocxLoader, ExcelLoader, PdfLoader
from mx_rag.document.splitter import CharTextSplitter
from mx_rag.retrievers.tree_retriever import split_text
from mx_rag.retrievers.tree_retriever.utils import _cal_chunks_when_exceed_max_tokens, _distances_from_embeddings

DOC_PARSER_MAP = {
    ".docx": (DocxLoader, CharTextSplitter),
    ".xlsx": (ExcelLoader, CharTextSplitter),
    ".xls": (ExcelLoader, CharTextSplitter),
    ".csv": (ExcelLoader, CharTextSplitter),
    ".pdf": (PdfLoader, CharTextSplitter),
}


class TestTokenParseDocumentFile(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    @staticmethod
    def token_parse_doucument_file(filepath, tokenizer, max_tokens, DOC_PARSER_MAP) -> Tuple[
        List[str], List[Dict[str, str]]]:
        file = Path(filepath)
        loader, splitter = DOC_PARSER_MAP.get(file.suffix, (None, None))
        if loader is None:
            raise ValueError(f"{file.suffix} is not support")
        metadatas = []
        texts = []
        for doc in loader(file.as_posix()).load():
            split_texts = ["test_txt"]
            metadatas.extend(doc.metadata for _ in split_texts)
            texts.extend(split_texts)
        return texts, metadatas

    def setUp(self):
        self.file_path = os.path.join(TestTokenParseDocumentFile.current_dir, "../../../data/demo.docx")

    def test_token_parse_doucument_file_unsupported_file_type(self):
        with self.assertRaises(ValueError):
            TestTokenParseDocumentFile.token_parse_doucument_file(
                os.path.join(TestTokenParseDocumentFile.current_dir, "../../../data/Sample.img"),
                None, 100, DOC_PARSER_MAP)

    def test_token_parse_doucument_file_sample(self):
        tokenizer = None
        texts, metadatas = TestTokenParseDocumentFile.token_parse_doucument_file(self.file_path, tokenizer, 100,
                                                                                 DOC_PARSER_MAP)
        self.assertEqual(metadatas, [{'source': 'demo.docx'}])

    def test_split_text(self):
        tokenizer = Mock()
        tokenizer.encode = MagicMock(return_value=[1, 2])
        result = split_text("this is a test txt", tokenizer, 5)
        self.assertEqual(['this is a test txt'], result)


    def test_split_text_split(self):
        tokenizer = Mock()
        tokenizer.encode = MagicMock(return_value=[1, 2])
        result = split_text("this is a？ test txt, split？ chunks", tokenizer, 3)
        self.assertEqual(['this is a', ' test txt, split', ' chunks'], result)

    def test_cal_chunks_when_exceed_max_tokens(self):
        tokenizer = Mock()
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        chunks = ["this is a test chunk"]
        _cal_chunks_when_exceed_max_tokens(chunks, 3, 0, "test sentence", tokenizer)
        self.assertEqual(['this is a test chunk', 'test sentence'], chunks)


    def test_distances_from_embeddings_not_in_metrics(self):
        with self.assertRaises(ValueError):
            _distances_from_embeddings([2.2], [[1.1]], "sine")