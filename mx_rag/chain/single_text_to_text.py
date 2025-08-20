# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import copy
from typing import Union, Iterator, List, Dict

from langchain_core.documents import Document
from loguru import logger

from langchain_core.retrievers import BaseRetriever

from mx_rag.utils.common import validate_params, BOOL_TYPE_CHECK_TIP, TEXT_MAX_LEN
from mx_rag.llm.llm_parameter import LLMParameterConfig
from mx_rag.chain import Chain
from mx_rag.llm import Text2TextLLM
from mx_rag.reranker.reranker import Reranker
from mx_rag.utils.common import MAX_PROMPT_LENGTH

DEFAULT_RAG_PROMPT = """根据上述已知信息，简洁和专业地回答用户的问题。如果无法从已知信息中得到答案，请根据自身经验做出回答"""
TEXT_RAG_TEMPLATE = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer as concise and accurate as possible.
        Do NOT repeat the question or output any other words.
        Context: {context} 
        Question: {question} 
        Answer:
"""


class SingleText2TextChain(Chain):
    document_separator: str = "\n\n"

    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM), message="param must be instance of Text2TextLLM"),
        retriever=dict(validator=lambda x: isinstance(x, BaseRetriever),
                       message="param must be instance of BaseRetriever"),
        reranker=dict(validator=lambda x: isinstance(x, Reranker) or x is None,
                      message="param must be None or instance of Reranker"),
        prompt=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= MAX_PROMPT_LENGTH,
                    message=f"param must be str and length range [1, {MAX_PROMPT_LENGTH}]"),
        source=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
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
        self._prompt = prompt
        self._source = source
        self._role: str = "user"

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
                  message=f"param must be a str and its length meets (0, {TEXT_MAX_LEN}]"),
        llm_config=dict(validator=lambda x: isinstance(x, LLMParameterConfig),
                        message="llm_config must be instance of LLMParameterConfig")
    )
    def query(self, text: str, llm_config: LLMParameterConfig = LLMParameterConfig(temperature=0.5, top_p=0.95),
              *args, **kwargs) \
            -> Union[Dict, Iterator[Dict]]:
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
               question: str,
               llm_config: LLMParameterConfig) -> Union[Dict, Iterator[Dict]]:

        q_docs = self._retriever.invoke(question)

        if self._reranker is not None and len(q_docs) > 0:
            scores = self._reranker.rerank(question, [doc.page_content for doc in q_docs])
            q_docs = self._reranker.rerank_top_k(q_docs, scores)

        q_with_prompt = self._merge_query_prompt(question, copy.deepcopy(q_docs), self._prompt)

        if not llm_config.stream:
            return self._do_query(q_with_prompt, llm_config, question=question, q_docs=q_docs)

        return self._do_stream_query(q_with_prompt, llm_config, question=question, q_docs=q_docs)

    def _do_query(self, q_with_prompt: str, llm_config: LLMParameterConfig, question: str, q_docs: List[Document]) \
            -> Dict:
        logger.info("invoke normal query")
        resp = {"query": question, "result": ""}
        if self._source:
            resp['source_documents'] = [{'metadata': x.metadata, 'page_content': x.page_content} for x in q_docs]
        llm_response = self._llm.chat(query=q_with_prompt, role=self._role, llm_config=llm_config)
        resp['result'] = llm_response
        return resp

    def _do_stream_query(self, q_with_prompt: str, llm_config: LLMParameterConfig, question: str,
                         q_docs: List[Document] = None) -> Iterator[Dict]:
        logger.info("invoke stream query")
        resp = {"query": question, "result": ""}
        if self._source and q_docs:
            resp['source_documents'] = [{'metadata': x.metadata, 'page_content': x.page_content} for x in q_docs]

        for response in self._llm.chat_streamly(query=q_with_prompt, role=self._role, llm_config=llm_config):
            resp['result'] = response
            yield resp


class GraphRagText2TextChain(SingleText2TextChain):
    def _query(self,
               question: str,
               llm_config: LLMParameterConfig) -> Union[Dict, Iterator[Dict]]:
        contexts = self._retriever.invoke(question)
        if self._reranker is not None and len(contexts) > 0:
            scores = self._reranker.rerank(question, contexts)
            contexts = self._reranker.rerank_top_k(contexts, scores)
        input_context = '\n'.join(contexts) if contexts else ""
        prompt = TEXT_RAG_TEMPLATE.format(context=input_context, question=question)
        if self._llm.llm_config.stream:
            return self._do_stream_query(prompt, llm_config, question, [])
        return self._do_query(prompt, llm_config, question, [])

    def _do_query(self, q_with_prompt: str, llm_config: LLMParameterConfig, question: str, q_docs: List[Document]) \
            -> Dict:
        logger.info("invoke normal query")
        resp = {"query": question, "result": ""}
        llm_response = self._llm.chat(query=q_with_prompt, role=self._role, llm_config=llm_config)
        resp['result'] = llm_response
        return resp

    def _do_stream_query(self, q_with_prompt: str, llm_config: LLMParameterConfig, question: str,
                         q_docs: List[Document] = None) -> Iterator[Dict]:
        logger.info("invoke stream query")
        resp = {"query": question, "result": ""}
        for response in self._llm.chat_streamly(query=q_with_prompt, role=self._role, llm_config=llm_config):
            resp['result'] = response
            yield resp
