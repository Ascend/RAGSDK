# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import List, Callable, Union, Dict

from langchain_core.pydantic_v1 import Field
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from loguru import logger
import numpy as np

from langchain_core.retrievers import BaseRetriever
from mx_rag.storage.document_store import Docstore
from mx_rag.storage.vectorstore import VectorStore
from mx_rag.utils.common import MAX_TOP_K, TEXT_MAX_LEN, validate_params


class Retriever(BaseRetriever):
    vector_store: VectorStore
    document_store: Docstore
    embed_func: Callable[[List[str]], Union[List[List[float]], List[Dict[int, float]]]]
    k: int = Field(default=1, ge=1, le=MAX_TOP_K)
    score_threshold: float = Field(default=None, ge=0.0, le=1.0)

    class Config:
        arbitrary_types_allowed = True

    @validate_params(
        query=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
                   message=f"query must be a str and length range (0, {TEXT_MAX_LEN}]")
    )
    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        embeddings = self.embed_func([query])

        if self.score_threshold is None:
            scores, indices = self.vector_store.search(embeddings, k=self.k)[:2]
        else:
            scores, indices = self.vector_store.search_with_threshold(embeddings, k=self.k,
                                                                      threshold=self.score_threshold)

        result = []

        for i, idx in enumerate(indices[0]):
            logger.debug(f"check {i}/{idx}")
            if idx < 0:
                continue
            doc = self.document_store.search(idx)
            if doc is None:
                continue
            logger.debug(f"scores {scores[0][i]}, page content len: {len(doc.page_content)}")
            result.append((doc, scores[0][i]))

        return self._post_process_result(result)

    def _post_process_result(self, result: List[tuple]):
        docs = [Document(page_content=doc.page_content, metadata=doc.metadata)
                for doc, similarity in result]

        if len(docs) == 0:
            logger.warning("no relevant documents found!!!")

        return docs[:self.k]
