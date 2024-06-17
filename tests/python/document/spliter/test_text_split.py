import unittest

from langchain.text_splitter import CharacterTextSplitter

from mx_rag.document.splitter.text_splitter import TextSplitterBase
from mx_rag.document.splitter import CharTextSplitter


class TestTextSplit(unittest.TestCase):
    def test_spliter_init_failed_case_1(self):
        try:
            CharTextSplitter(chunk_size=-1, chunk_overlap=4, separator="\n")
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_spliter_init_failed_case_2(self):
        try:
            CharTextSplitter(chunk_size=10, chunk_overlap=-1, separator="\n")
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_spliter_init_failed_case_3(self):
        try:
            CharTextSplitter(chunk_size=1, chunk_overlap=10, separator="\n")
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_blank_space_odd_num_text_split(self):
        text = "123 \n 456 \n 789"
        text_splitter = TextSplitterBase(chunk_size=512, chunk_overlap=100, keep_separator=True)
        split_text = text_splitter._split_text(text, "\n")
        expect_split_text = ["123 ", "\n 456 ", "\n 789"]
        self.assertEqual(expect_split_text, split_text)

    def test_blank_space_even_num_text_split(self):
        text = "123 \n 456 \n 789 \n"
        text_splitter = TextSplitterBase(chunk_size=512, chunk_overlap=100, keep_separator=True)
        split_text = text_splitter._split_text(text, "\n")
        expect_split_text = ["123 ", "\n 456 ", "\n 789 ", "\n"]
        self.assertEqual(expect_split_text, split_text)

    def test_char_split(self):
        text = "123456789ABCDEFG \n 456 \n 1 \n 789 \n"
        langchain_char_text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=4, separator="\n")
        expect_text = langchain_char_text_splitter.split_text(text)

        mxrag_char_text_splitter = CharTextSplitter(chunk_size=10, chunk_overlap=4, separator="\n")
        result_text = mxrag_char_text_splitter.split_text(text)
        self.assertEqual(expect_text, result_text)
