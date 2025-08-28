# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from typing import List, Union, Dict
from loguru import logger

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, ConfigDict

from mx_rag.utils.common import TEXT_MAX_LEN, validate_params, MAX_TOP_K, MAX_FILTER_SEARCH_ITEM, MAX_STDOUT_STR_LEN
from mx_rag.storage.document_store import MilvusDocstore, OpenGaussDocstore


class FullTextRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    document_store: Union[MilvusDocstore, OpenGaussDocstore]
    k: int = Field(default=1, ge=1, le=MAX_TOP_K)
    filter_dict: dict = {}

    @validate_params(
        query=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
                   message=f"query must be a str and length range (0, {TEXT_MAX_LEN}]")
    )
    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:

        docs = self.document_store.full_text_search(query, top_k=self.k, filter_dict=self.filter_dict)
        result = []
        for doc in docs:
            metadata = doc.metadata
            metadata.update({'retrieval_type': 'sparse'})
            result.append(Document(page_content=doc.page_content, metadata=doc.metadata))

        if not result:
            logger.warning("no relevant documents found!!!")

        return result

    @validate_params(
        filter_dict=dict(validator=lambda x: isinstance(x, Dict) and 0 < len(x) <= MAX_FILTER_SEARCH_ITEM,
                         message=f"filter_dict must be a dict and length range (0, {MAX_FILTER_SEARCH_ITEM}]")
    )
    def set_filter(self, filter_dict: dict):
        invalid_keys = str(filter_dict.keys() - {"document_id"})
        if invalid_keys:
            logger.warning(f"{invalid_keys[:MAX_STDOUT_STR_LEN]} ... is no support")
        self.filter_dict = filter_dict