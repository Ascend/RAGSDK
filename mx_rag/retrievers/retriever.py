# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from abc import ABC
from typing import List, Tuple

from loguru import logger

from mx_rag.document.loader.docx_loader import Doc
from mx_rag.storage import Document
from mx_rag.vectorstore.faiss_npu import MindFAISS


class Retriever(ABC):

    def __init__(self, vector_store: MindFAISS, embed_func, k: int = 1, score_threshold: float = 0.1):
        super().__init__()
        self._vector_store = vector_store
        self._embed_func = embed_func
        self._k = k
        self._score_threshold = score_threshold
        self._ref_doc = []

    @property
    def ref_doc(self) -> str:
        if len(self._ref_doc) != 0:
            return "。参考资料:" + '\n'.join(x.metadata['filepath'] for x in self._ref_doc)
        return ""

    def get_relevant_documents(self, query: str) -> List[Doc]:
        docs = self._get_relevant_documents(query)
        self._ref_doc = docs[:self._k]
        return docs[:self._k]

    def _get_relevant_documents(self, query: str) -> List[Doc]:
        sr: List[Tuple[Document, float]] = self._vector_store.similarity_search([query], self._embed_func, k=self._k)
        logger.info(f"Filter is [<={self._score_threshold}]")
        docs = [Doc(doc.page_content, doc.metadata) for doc, similarity in sr if similarity <= self._score_threshold]

        if len(docs) == 0:
            logger.warning("no relevant documents found!!!")

        return docs
