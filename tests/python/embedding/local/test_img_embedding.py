import os
import json
import unittest
from unittest.mock import patch

from mx_rag.embedding.local import ImageEmbedding
from mx_rag.utils.common import IMG_EMBBEDDING_TEXT_LEN


class TestImageEmbedding(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        os.mkdir("/tmp/chinese-clip-vit-base-patch16/")
        data = {"vision_config": {'image_size': 224}}
        with open("/tmp/chinese-clip-vit-base-patch16/config.json", "w") as fi:
            fi.write(json.dumps(data))

    @classmethod
    def teardown_class(cls):
        os.remove("/tmp/chinese-clip-vit-base-patch16/config.json")
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
        with self.assertRaises(ValueError):
            ret = emb.embed_documents([])

        text = ["a"] * 1000001
        with self.assertRaises(ValueError):
            ret = emb.embed_documents(text)

    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoProcessor.from_pretrained")
    def test_embed_query_para_invalid(self,
                                      processor_pre_mock,
                                      model_pre_mock):
        model_pre_mock.return_value = None
        processor_pre_mock.return_value = None

        emb = ImageEmbedding(model_path="/tmp/chinese-clip-vit-base-patch16/", dev_id=3, use_fp16=False)
        with self.assertRaises(ValueError):
            emb.embed_query("")

    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoProcessor.from_pretrained")
    def test_embed_images_para_invalid(self,
                                       processor_pre_mock,
                                       model_pre_mock):
        model_pre_mock.return_value = None
        processor_pre_mock.return_value = None

        emb = ImageEmbedding(model_path="/tmp/chinese-clip-vit-base-patch16/", dev_id=3, use_fp16=False)
        with self.assertRaises(ValueError):
            emb.embed_images([])
        text = ["a"] * 1001
        with self.assertRaises(ValueError):
            emb.embed_images(text)


if __name__ == '__main__':
    unittest.main()
