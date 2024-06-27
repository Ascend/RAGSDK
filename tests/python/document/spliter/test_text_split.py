import os
import unittest

from langchain.text_splitter import CharacterTextSplitter

from mx_rag.document.splitter.text_splitter import TextSplitterBase
from mx_rag.document.splitter import CharTextSplitter
from mx_rag.document.loader import DocxLoader
from mx_rag.document.loader import PdfLoader
from mx_rag.document.loader import ExcelLoader


class TestTextSplit(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    def test_spliter_sec_check_case(self):
        illegal_param_list = [
            {"chunk_size": -1, "chunk_overlap": 4},  # chunk_size < 0
            {"chunk_size": None, "chunk_overlap": 4},  # chunk_size = None
            {"chunk_size": 10, "chunk_overlap": -1},  # chunk_overlap < 0
            {"chunk_size": 10, "chunk_overlap": None},  # chunk_overlap = None
            {"chunk_size": 1, "chunk_overlap": 10},  # chunk_size < chunk_overlap
            {"chunk_size": 40.6, "chunk_overlap": 10},  # chunk_size is float
            {"chunk_size": 40, "chunk_overlap": 10.2},  # chunk_overlap is float
            {"chunk_size": "hello world", "chunk_overlap": 10},  # chunk_size is string
            {"chunk_size": 40, "chunk_overlap": "hello world"},  # chunk_overlap is string
        ]

        for param in illegal_param_list:
            try:
                CharTextSplitter(chunk_size=param["chunk_size"], chunk_overlap=param["chunk_overlap"], separator="\n")
            except ValueError:
                self.assertTrue(True)
            else:
                self.assertTrue(False)

    def test_text_split_case(self):
        text_param_list = [
            {
                "text": "我喜欢在阳光明媚的日子里散步，我也喜欢在雨天打乒乓球和篮球;冰淇凌好吃;他喜欢游泳;小明喜欢在深夜大声唱山歌",
                "chunk_size": 40, "chunk_overlap": 10, "separator": ";"},
            {
                "text": "我喜欢在阳光明媚的日子里散步，我也喜欢在雨天打乒乓球和篮球;冰淇凌好吃;小明喜欢在深夜大声唱山歌",
                "chunk_size": 40, "chunk_overlap": 10, "separator": ";"},
            {
                "text": "我喜欢在阳光明媚的日子里散步，呼吸新鲜空气，我也喜欢在雨天打乒乓球和篮球;小明喜欢在深夜大声唱山歌",
                "chunk_size": 40, "chunk_overlap": 10, "separator": ";"},
            {
                "text": "我喜欢在阳光明媚的日子里散步，呼吸新鲜空气;我也喜欢在雨天散步",
                "chunk_size": 30, "chunk_overlap": 10, "separator": ";"},
            {
                "text": "她们喜欢晴天骑自行车;他喜欢游泳;冰淇淋好吃;我喜欢在阳光明媚的日子里散步，呼吸新鲜空气;他喜欢晴天游泳;我也喜欢在雨天打乒乓球和篮球",
                "chunk_size": 40, "chunk_overlap": 10, "separator": ";"},
            {
                "text": "她们喜欢晴天骑自行车;我喜欢在阳光明媚的日子里散步，呼吸新鲜空气;他喜欢晴天游泳;我也喜欢在雨天打乒乓球和篮球",
                "chunk_size": 40, "chunk_overlap": 10, "separator": ";"},
            {
                "text": "我喜欢在阳光明媚的日子里散步，呼吸新鲜空气;我也喜欢在雨天打乒乓球和篮球",
                "chunk_size": 30, "chunk_overlap": 10, "separator": ";"},
            {
                "text": "我喜欢在阳光明媚的日子里散步，呼吸新鲜空气，感受大自然的美好，这让我心情舒畅",
                "chunk_size": 30, "chunk_overlap": 10, "separator": ";"}
        ]

        for param in text_param_list:
            self.compare_with_gloden(param["text"], CharTextSplitter, CharacterTextSplitter,
                                     param["chunk_size"], param["chunk_overlap"], param["separator"])

    def test_document_load_and_splite_case(self):
        test_doc_list = [
            {"doc": "../../../data/demo.docx", "loader": DocxLoader},  # english docx
            {"doc": "../../../data/mxVision.docx", "loader": DocxLoader},  # chinese docx
            {"doc": "../../../data/test.pdf", "loader": PdfLoader},  # english pdf
            {"doc": "../../../data/test_cn.pdf", "loader": PdfLoader},  # chinese pdf
            {"doc": "../../../data/test.xlsx", "loader": ExcelLoader}  # chinese mix english excel
        ]

        test_split_param_list = [
            {"chunk_size": 1000, "chunk_overlap": 0, "separator": ":"},
            {"chunk_size": 1000, "chunk_overlap": 0, "separator": ";"},
            {"chunk_size": 1000, "chunk_overlap": 0, "separator": "?"},
            {"chunk_size": 1000, "chunk_overlap": 0, "separator": ''},
            {"chunk_size": 1000, "chunk_overlap": 0, "separator": "z"},
            {"chunk_size": 1000, "chunk_overlap": 0, "separator": "|"},
            {"chunk_size": 1000, "chunk_overlap": 1000, "separator": ":"},
            {"chunk_size": 1000, "chunk_overlap": 1000, "separator": ";"},
            {"chunk_size": 1000, "chunk_overlap": 1000, "separator": "?"},
            {"chunk_size": 1000, "chunk_overlap": 1000, "separator": ''},
            {"chunk_size": 1000, "chunk_overlap": 1000, "separator": "z"},
            {"chunk_size": 1000, "chunk_overlap": 1000, "separator": "|"},
            {"chunk_size": 1000, "chunk_overlap": 100, "separator": ":"},
            {"chunk_size": 1000, "chunk_overlap": 100, "separator": ";"},
            {"chunk_size": 1000, "chunk_overlap": 100, "separator": "?"},
            {"chunk_size": 1000, "chunk_overlap": 100, "separator": ''},
            {"chunk_size": 1000, "chunk_overlap": 100, "separator": "z"},
            {"chunk_size": 1000, "chunk_overlap": 100, "separator": "|"},
        ]

        for doc in test_doc_list:
            doc_path = os.path.join(self.current_dir, doc["doc"])
            load_metadata = doc["loader"](doc_path).load()
            doc_text = load_metadata[0].page_content

            for split_param in test_split_param_list:
                self.compare_with_gloden(doc_text,
                                         CharTextSplitter,
                                         CharacterTextSplitter,
                                         chunk_size=split_param["chunk_size"],
                                         chunk_overlap=split_param["chunk_overlap"],
                                         separator=split_param["separator"])

    def compare_with_gloden(self, text, mx_rag_spliter, gloden_spliter, chunk_size, chunk_overlap, separator):
        langchain_char_text_splitter = gloden_spliter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                      separator=separator)
        expect_text = langchain_char_text_splitter.split_text(text)

        mxrag_char_text_splitter = mx_rag_spliter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                  separator=separator)
        result_text = mxrag_char_text_splitter.split_text(text)
        self.assertEqual(expect_text, result_text)
