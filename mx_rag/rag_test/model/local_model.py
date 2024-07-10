# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import List

from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.run_config import RunConfig

from mx_rag.embedding.local.text_embedding import TextEmbedding


class LocalEmbedding(BaseRagasEmbeddings):

    def __init__(self, path: str, max_length=512, batch_size=256):
        self.model = TextEmbedding(path)
        self.max_length = max_length
        self.batch_size = batch_size
        self.set_run_config(RunConfig())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_texts(texts, self.batch_size, self.max_length).tolist()

    def embed_query(self, text: str) -> List[float]:
        result = self.model.embed_texts([text], self.batch_size, self.max_length).tolist()
        return result[0] if result else []
