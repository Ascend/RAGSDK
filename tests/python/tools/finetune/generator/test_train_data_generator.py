#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import unittest
from unittest.mock import patch

import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from mx_rag.document import LoaderMng
from mx_rag.document.loader import DocxLoader
from mx_rag.llm import Text2TextLLM
from mx_rag.reranker.local import LocalReranker
from mx_rag.tools.finetune.generator import TrainDataGenerator
from mx_rag.utils import ClientParam


class TestTrainDataGenerator(unittest.TestCase):
    @patch("mx_rag.reranker.local.LocalReranker.__init__")
    @patch("mx_rag.reranker.local.LocalReranker.rerank")
    def setUp(self, fake_rerank, fake_init):
        def f_reranker(query: str, texts: list[str]):
            return np.array([1] * len(texts))

        fake_init.return_value = None
        fake_rerank.side_effect = f_reranker

        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.file_path = os.path.realpath(os.path.join(current_dir, "../../../../data/"))
        client_param = ClientParam(use_http=True, timeout=120)
        llm = Text2TextLLM(model_name="test_model_name", base_url="test_url", client_param=client_param)
        rerank = LocalReranker(model_path='/model/reranker')
        self.train_data_generator = TrainDataGenerator(llm, self.file_path, rerank)

    @patch("mx_rag.utils.file_check.FileCheck.dir_check")
    def test_generate_origin_document(self, dir_check_mock):
        loader_mng = LoaderMng()

        loader_mng.register_loader(loader_class=DocxLoader, file_types=[".docx"])

        # 加载文档切分器，使用langchain的
        loader_mng.register_splitter(splitter_class=RecursiveCharacterTextSplitter,
                                     file_types=[".docx"],
                                     splitter_params={"chunk_size": 750,
                                                      "chunk_overlap": 150,
                                                      "keep_separator": False
                                                      }
                                     )
        result = self.train_data_generator.generate_origin_document(os.path.join(self.file_path, "files"), loader_mng)
        self.assertNotEqual([], result)
