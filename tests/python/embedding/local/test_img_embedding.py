import os
import unittest
from unittest.mock import patch

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
    def test_embed_documents_para_invalid(self,
                                          processor_pre_mock,
                                          model_pre_mock):
        model_pre_mock.return_value = None
        processor_pre_mock.return_value = None

        emb = ImageEmbedding(model_path="/tmp/chinese-clip-vit-base-patch16/", dev_id=3, use_fp16=False)
        try:
            ret = emb.embed_documents([])
        except Exception as e:
            self.assertEqual(f"{e}", "texts length equal 0")

        text = ["a"] * 1000001
        try:
            ret = emb.embed_documents(text)
        except Exception as e:
            self.assertEqual(f"{e}", f'texts length greater than{emb.TEXT_COUNT}')

    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoProcessor.from_pretrained")
    def test_embed_query_para_invalid(self,
                                      processor_pre_mock,
                                      model_pre_mock):
        model_pre_mock.return_value = None
        processor_pre_mock.return_value = None

        emb = ImageEmbedding(model_path="/tmp/chinese-clip-vit-base-patch16/", dev_id=3, use_fp16=False)
        try:
            ret = emb.embed_query("")
        except Exception as e:
            self.assertEqual(f"{e}", f"the length of text in texts greater than {emb.TEXT_LEN} or equal 0")


    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoProcessor.from_pretrained")
    def test_embed_images_para_invalid(self,
                                       processor_pre_mock,
                                       model_pre_mock):
        model_pre_mock.return_value = None
        processor_pre_mock.return_value = None

        emb = ImageEmbedding(model_path="/tmp/chinese-clip-vit-base-patch16/", dev_id=3, use_fp16=False)
        try:
            ret = emb.embed_images([])
        except Exception as e:
            self.assertEqual(f"{e}", "images length equal 0")

        text = ["a"] * 1001
        try:
            ret = emb.embed_images(text)
        except Exception as e:
            self.assertEqual(f"{e}", f'images length greater than {emb.IMAGE_COUNT}')