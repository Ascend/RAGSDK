
import os
import unittest
from unittest.mock import patch, MagicMock

from mx_rag.document.loader.image_loader import ImageLoader

class ImageLoaderTestCase(unittest.TestCase):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.realpath(os.path.join(current_dir, "../../../data"))

    def test_lazy_load(self):
        loader = ImageLoader(os.path.join(self.data_dir, "test.png"))
        d = loader.lazy_load()
        self.assertTrue(hasattr(d, '__iter__'), "lazy_load 应返回一个迭代器")
        self.assertTrue(hasattr(d, '__next__'), "lazy_load 应返回一个迭代器")

    def test_load(self):
        loader = ImageLoader(os.path.join(self.data_dir, "test.png"))
        png = loader.load()
        self.assertTrue(png[0].metadata, {"path": os.path.join(self.current_dir, "../../../data/test.png")})

