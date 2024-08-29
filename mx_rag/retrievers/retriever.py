# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import List, Callable

from langchain_core.pydantic_v1 import validator, Field
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from loguru import logger
import numpy as np


from langchain_core.retrievers import BaseRetriever
from mx_rag.storage.document_store import Docstore
from mx_rag.storage.vectorstore import VectorStore
from mx_rag.utils.common import MAX_TOP_K


class Retriever(BaseRetriever):

    vector_store: VectorStore
    document_store: Docstore
    embed_func: Callable[[List[str]], List[List[float]]]
    k: int = Field(default=1, ge=1, le=MAX_TOP_K)
    score_threshold: float = Field(default=0.1, ge=0.0)


    class Config:
        arbitrary_types_allowed = True


    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        embedding = self.embed_func([query])
        scores, indices = self.vector_store.search(np.array(embedding), k=self.k)
        sr = []

        for i, idx in enumerate(indices[0]):
            logger.debug(f"check {i}/{idx}")
            doc = self.document_store.search(idx)
            if doc is None:
                continue
            logger.debug(f"scores {scores[0][i]}, page content len: {len(doc.page_content)}")
            sr.append((doc, scores[0][i]))

        logger.info(f"Filter is [<={self.score_threshold}]")
        docs = [Document(page_content=doc.page_content, metadata=doc.metadata)
                for doc, similarity in sr
                if similarity <= self.score_threshold]

        if len(docs) == 0:
            logger.warning("no relevant documents found!!!")

        return docs[:self.k]
