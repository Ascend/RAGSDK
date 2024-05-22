import unittest

from mx_rag.document.loader import DocxLoader


class DocxLoaderTestCase(unittest.TestCase):
    def test_load(self):
        loader = DocxLoader("../../../data/demo.docx")
        d = loader.load()
        self.assertEqual(1, len(d))

    def test_load_with_image(self):
        loader = DocxLoader("../../../data/demo.docx", image_inline=True)
        d = loader.load()
        self.assertEqual(1, len(d))


if __name__ == '__main__':
    unittest.main()
