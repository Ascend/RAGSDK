# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from abc import ABC
from typing import Union, Iterator, List, Dict

from loguru import logger

DEFAULT_RAG_PROMPT = (
    "<指令>根据上述已知信息，简洁和专业的来回答用户的问题，并保存对应知识的URL。"
    "如果无法从中已知信息中得到答案，请在事先声明 “没有搜索到足够的相关信息，"
    "以下是根据我的经验做出的的回答”后，根据先前训练过的知识回答问题，"
    "不需要提到提供的知识片段。</指令>"
)


class SimpleRetrieval(ABC):
    def __init__(self, retriever, llm, prompt=DEFAULT_RAG_PROMPT):
        super().__init__()
        self._retriever = retriever
        self._llm = llm
        self._content = ""
        self._prompt = prompt
        self._source = False
        self._history: List[Dict] = []
        self._role: str = ""
        logger.debug(f"RAG prompt: {self._prompt}")

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value: bool):
        self._source = value

    def query(self,
              text: str,
              max_tokens: int,
              temperature: float,
              top_p: float,
              stream: bool = False) -> Union[str, Iterator]:
        return self._query(text, max_tokens, temperature, top_p, stream)

    def _query(self,
               text: str,
               max_tokens: int,
               temperature: float,
               top_p: float,
               stream: bool = False) -> Union[str, Iterator]:

        prompt = self._retriever.get_relevant_documents(text, self._prompt)
        logger.debug(f"query prompt: {prompt}")

        if not stream:
            return self._do_query(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

        return self._do_stream_query(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def _do_query(self, text, **kwargs) -> str:
        logger.info("invoke normal query")
        llm_response = self._llm.chat(text, self._history, self._role, **kwargs)
        self._content = llm_response
        if self.source:
            llm_response += self._retriever.ref_doc
        return self._content

    def _do_stream_query(self, text, **kwargs) -> Iterator:
        logger.info("invoke stream query")
        for resp in self._llm.chat_streamly(text, self._history, self._role, **kwargs):
            self._content = resp
            yield self._content

        if self.source:
            yield self._retriever.ref_doc
