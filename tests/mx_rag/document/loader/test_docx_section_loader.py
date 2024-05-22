import unittest

from langchain.text_splitter import RecursiveCharacterTextSplitter

from mx_rag.document.loader.docx_section_loader import *


class DocxSectionLoaderTestCase(unittest.TestCase):
    def test_load(self):
        loader = DocxLoaderByHead("../../../data/demo.docx")
        res = loader.load()
        self.assertEqual(1, len(res))

    def test_load_and_split(self):
        loader = DocxLoaderByHead("../../../data/demo.docx")
        res = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100))
        self.assertEqual(2, len(res))

    def test_title(self):
        loader = DocxLoaderByHead("../../../data/title.docx")
        res = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100))
        self.assertEqual(1, len(res))

    def test_link(self):
        loader = DocxLoaderByHead("../../../data/link.docx")
        res = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100))
        self.assertEqual(7, len(res))


if __name__ == '__main__':
    unittest.main()
