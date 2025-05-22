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
    filter_dict: dict = {}

    class Config:
        arbitrary_types_allowed = True

    @validate_params(
        query=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
                   message=f"query must be a str and length range (0, {TEXT_MAX_LEN}]")
    )
    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        embeddings = self.embed_func([query])
        # 只有MindFAISS没有search_mode
        if not hasattr(self.vector_store, "search_mode"):
            embeddings = np.array(embeddings)

        if self.score_threshold is None:
            scores, indices = self.vector_store.search(embeddings, k=self.k, filter_dict=self.filter_dict)[:2]
        else:
            scores, indices = self.vector_store.search_with_threshold(embeddings, k=self.k,
                                                                      threshold=self.score_threshold,
                                                                      filter_dict=self.filter_dict)
        result = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                logger.warning(f"Index error: idx expected non-negative, given {idx}")
                continue
            doc = self.document_store.search(idx)
            if doc:
                result.append((doc, score))
        return self._post_process_result(result)

    def set_filter(self, filter_dict: dict):
        self.filter_dict = filter_dict

    def _post_process_result(self, result: List[tuple]):
        docs = []
        for doc, score in result:
            metadata = doc.metadata
            metadata.update({'score': score})
            docs.append(Document(page_content=doc.page_content, metadata=doc.metadata))
        if not docs:
            logger.warning("no relevant documents found!!!")
        return docs[:self.k]