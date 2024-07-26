# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import List

from mx_rag.reranker.reranker import Reranker
from mx_rag.storage.vectorstore import VectorStore


class MockerReranker(Reranker):
    def rerank(self, query: str, texts: List[str], batch_size: int = 1):
        return [1] if query.lower() == texts[0].lower() else [0]


class MockerVecStorage(VectorStore):
    def delete(self, ids):
        pass

    def search(self, embeddings, k):
        pass

    def add(self, embeddings, ids):
        pass

    def get_ntotal(self) -> int:
        return 10
