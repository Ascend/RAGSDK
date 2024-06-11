import os
import unittest
from unittest.mock import patch

import numpy as np

from mx_rag.embedding.local import ImageEmbedding


class TestImageEmbedding(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        os.mkdir("/tmp/chinese-clip-vit-base-patch16/")
    @classmethod
    def teardown_class(cls):
        os.removedirs("/tmp/chinese-clip-vit-base-patch16/")

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoProcessor.from_pretrained")
    def test_embed_texts_para_invalid(self,
                                      model_pre_mock,
                                      processor_pre_mock,):
        model_pre_mock.return_value = None
        processor_pre_mock.return_value = None

        emb = ImageEmbedding(model_path="/tmp/chinese-clip-vit-base-patch16/", dev_id=3, use_fp16=False)
        ret = emb.embed_texts([])
        self.assertEqual(ret.size, 0)

        text = ["a"] * 1001
        ret = emb.embed_texts(text)
        self.assertEqual(ret.size, 0)


    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoProcessor.from_pretrained")
    def test_embed_images_para_invalid(self,
                                      model_pre_mock,
                                      processor_pre_mock,):
        model_pre_mock.return_value = None
        processor_pre_mock.return_value = None

        emb = ImageEmbedding(model_path="/tmp/chinese-clip-vit-base-patch16/", dev_id=3, use_fp16=False)
        ret = emb.embed_images([])
        self.assertEqual(ret.size, 0)

        text = ["a"] * 1001
        ret = emb.embed_images(text)
        self.assertEqual(ret.size, 0)