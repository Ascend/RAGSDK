# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from abc import ABC
from typing import List

from langchain_core.documents import Document
from loguru import logger

from mx_rag.storage.document_store import Docstore


class Retriever(ABC):

    def __init__(self, vector_store, document_store: Docstore, embed_func, k: int = 1, score_threshold: float = 0.1):
        super().__init__()
        self._vector_store = vector_store
        self._document_store = document_store
        self._embed_func = embed_func
        self._k = k
        self._score_threshold = score_threshold

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self._get_relevant_documents(query)
        return docs[:self._k]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        embedding = self._embed_func([query])
        scores, indices = self._vector_store.search(embedding, k=self._k)
        sr = []

        for i, idx in enumerate(indices[0]):
            logger.debug(f"check {i}/{idx}")
            doc = self._document_store.search(idx)
            if doc is None:
                continue
            logger.debug(f"scores {scores[0][i]}, page content len: {len(doc.page_content)}")
            sr.append((doc, scores[0][i]))

        logger.info(f"Filter is [<={self._score_threshold}]")
        docs = [Document(page_content=doc.page_content, metadata=doc.metadata)
                for doc, similarity in sr
                if similarity <= self._score_threshold]

        if len(docs) == 0:
            logger.warning("no relevant documents found!!!")

        return docs
