# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import copy
from typing import Union, Iterator, List, Dict

from langchain_core.documents import Document
from loguru import logger

from langchain_core.retrievers import BaseRetriever

from mx_rag.utils.common import validate_params
from mx_rag.llm.llm_parameter import LLMParameterConfig
from mx_rag.chain import Chain
from mx_rag.llm import Text2TextLLM
from mx_rag.reranker.reranker import Reranker

DEFAULT_RAG_PROMPT = """根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中已知信息中得到答案，请根据自身经验做出的的回答"""


class SingleText2TextChain(Chain):
    document_separator: str = "\n\n"

    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM)),
        retriever=dict(validator=lambda x: isinstance(x, BaseRetriever) or x is None),
        reranker=dict(validator=lambda x: isinstance(x, Reranker) or x is None),
        prompt=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 512 * 1024),
        source=dict(validator=lambda x: isinstance(x, bool))
    )
    def __init__(self, llm: Text2TextLLM,
                 retriever: BaseRetriever,
                 reranker: Reranker = None,
                 prompt: str = DEFAULT_RAG_PROMPT,
                 source: bool = True):
        super().__init__()
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm
        self._content = ""
        self._prompt = prompt
        self._source = source
        self._history: List[Dict] = []
        self._role: str = "user"
        self._docs = []
        self._query_str = ""

    def query(self, text: str, llm_config: LLMParameterConfig = LLMParameterConfig(temperature=0.5, top_p=0.95),
              *args, **kwargs) \
            -> Union[Dict, Iterator[Dict]]:
        self._history = []
        return self._query(text, llm_config)

    def _merge_query_prompt(self, query: str, docs: List[Document], prompt: str):
        final_prompt = ""
        if len(docs) != 0:
            if prompt != "":
                last_doc = docs[-1]
                last_doc.page_content = (last_doc.page_content
                                         + f"{self.document_separator}{prompt}")
                docs[-1] = last_doc
            final_prompt = self.document_separator.join(x.page_content for x in docs)

        if final_prompt != "":
            final_prompt += self.document_separator

        final_prompt += query
        return final_prompt

    def _query(self,
               text: str,
               llm_config: LLMParameterConfig) -> Union[Dict, Iterator[Dict]]:

        self._query_str = text
        self._docs = self._retriever.invoke(text)

        if self._reranker is not None and len(self._docs) > 0:
            scores = self._reranker.rerank(text, [doc.page_content for doc in self._docs])
            self._docs = self._reranker.rerank_top_k(self._docs, scores)

        question = self._merge_query_prompt(text, copy.deepcopy(self._docs), self._prompt)

        if not llm_config.stream:
            return self._do_query(question, llm_config)

        return self._do_stream_query(question, llm_config)

    def _do_query(self, text: str, llm_config: LLMParameterConfig) -> Dict:
        logger.info("invoke normal query")
        resp = {"query": self._query_str, "result": ""}
        if self._source:
            resp['source_documents'] = [{'metadata': x.metadata, 'page_content': x.page_content} for x in self._docs]
        llm_response = self._llm.chat(text, self._history, self._role, llm_config)
        self._content = llm_response
        resp['result'] = llm_response
        return resp

    def _do_stream_query(self, text: str, llm_config: LLMParameterConfig) -> Iterator[Dict]:
        logger.info("invoke stream query")
        resp = {"query": self._query_str, "result": ""}
        if self._source:
            resp['source_documents'] = [{'metadata': x.metadata, 'page_content': x.page_content} for x in self._docs]

        for response in self._llm.chat_streamly(text, self._history, self._role, llm_config):
            self._content = response
            resp['result'] = response
            yield resp
