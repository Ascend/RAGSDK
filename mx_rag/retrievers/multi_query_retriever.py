# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import re
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from loguru import logger

from langchain_core.retrievers import BaseRetriever
from mx_rag.llm import Text2TextLLM
from mx_rag.retrievers.retriever import Retriever

DEFAULT_QUERY_PROMPT_EN = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is 
        to generate 3 different versions of the given user 
        question to retrieve relevant documents from a vector database. 
        Your goal is to help the user overcome some of the limitations 
        of distance-based similarity search. Provide these alternative 
        questions in english.Please number the answers starting from 1 
        and separated by newlines. Original question: {question}"""
)

DEFAULT_QUERY_PROMPT_CH = PromptTemplate(
    input_variables=["question"],
    template="""你是一个人工智能语言模型助理。您的任务是根据用户的原始问题，从矢量数据库中基于
    距离的相似性检索出与原问题相关的3个问题。你的目标是通过生成的多个角度的提问来帮助
    用户克服原问题的限制。请从1开始编号且用中文回答，每个回答用换行符分隔开。原始问题：{question}"""
)


class DefaultOutputParser(BaseOutputParser):
    @staticmethod
    def _is_starting_with_number(query: str):
        return bool(re.match(r'\d.*', query))

    def parse(self, text: str) -> List[str]:
        lines = []
        for line in text.splitlines():
            if self._is_starting_with_number(line.strip()):
                lines.append(line)
        return lines


class MultiQueryRetriever(Retriever, BaseRetriever):
    llm: Text2TextLLM
    prompt: PromptTemplate = DEFAULT_QUERY_PROMPT_CH
    parser: BaseOutputParser = DefaultOutputParser()
    max_tokens: int = 512

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        docs = []

        llm_query = self.prompt.format(question=query)
        llm_response = self.llm.chat(query=llm_query, history=[], role="user", max_tokens=self.max_tokens)
        for sub_query in self.parser.parse(text=str(llm_response)):
            logger.success(f"sub_query {sub_query}")
            doc = super(MultiQueryRetriever, self)._get_relevant_documents(sub_query)
            docs.extend(doc)

        docs = [doc for i, doc in enumerate(docs) if doc not in docs[:i]]
        return sorted(docs, key=lambda x: len(x.page_content))
