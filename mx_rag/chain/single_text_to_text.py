# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import copy
from typing import Union, Iterator, List, Dict

from loguru import logger

from mx_rag.chain import Chain
from mx_rag.document.doc import Doc

DEFAULT_RAG_PROMPT = (
    "<指令>根据上述已知信息，简洁和专业的来回答用户的问题，并保存对应知识的URL。"
    "如果无法从中已知信息中得到答案，请在事先声明 “没有搜索到足够的相关信息，"
    "以下是根据我的经验做出的的回答”后，根据先前训练过的知识回答问题，"
    "不需要提到提供的知识片段。</指令>"
)


class SingleText2TextChain(Chain):
    document_separator: str = "\n\n"

    def __init__(self, llm, retriever, prompt=DEFAULT_RAG_PROMPT):
        super().__init__()
        self._retriever = retriever
        self._llm = llm
        self._content = ""
        self._prompt = prompt
        self._source = False
        self._history: List[Dict] = []
        self._role: str = "user"
        self._doc = []
        self._query_str = ""

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value: bool):
        self._source = value

    def query(self, text: str, *args, **kwargs) -> Union[Dict, Iterator[Dict]]:
        return self._query(text, *args, **kwargs)

    def _merge_query_prompt(self, query: str, docs: List[Doc], prompt: str):
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
               max_tokens: int = 1000,
               temperature: float = 0.5,
               top_p: float = 0.95,
               stream: bool = False) -> Union[Dict, Iterator[Dict]]:

        self._query_str = text
        self._doc = self._retriever.get_relevant_documents(text)
        question = self._merge_query_prompt(text, copy.deepcopy(self._doc), self._prompt)
        logger.debug(f"query prompt: {self._prompt}")

        if not stream:
            return self._do_query(question, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

        return self._do_stream_query(question, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def _do_query(self, text: str, **kwargs) -> Dict:
        logger.info("invoke normal query")
        resp = {"query": self._query_str, "result": ""}
        if self.source:
            resp['source_documents'] = [vars(x) for x in self._doc]
        llm_response = self._llm.chat(text, self._history, self._role, **kwargs)
        self._content = llm_response
        resp['result'] = llm_response
        return resp

    def _do_stream_query(self, text: str, **kwargs) -> Iterator[Dict]:
        logger.info("invoke stream query")
        resp = {"query": self._query_str, "result": ""}
        if self.source:
            resp['source_documents'] = [vars(x) for x in self._doc]

        for response in self._llm.chat_streamly(text, self._history, self._role, **kwargs):
            self._content = response
            resp['result'] = response
            yield resp
