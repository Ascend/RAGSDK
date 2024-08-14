# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import re
from abc import abstractmethod, ABC
from typing import List, Any

from langchain_core.documents import Document
from loguru import logger

from mx_rag.llm import Text2TextLLM
from mx_rag.retrievers.retriever import Retriever


class PromptTemplate:
    def __init__(self, template_string):
        self._template_string = re.sub(r"\s+", " ", template_string)

    def format(self, **kwargs):
        """
        Format the template with the given keyword arguments.

        :param kwargs: Key-value pairs to replace in the template.
        :return: A string with the rendered template.
        """
        return self._template_string.format(**kwargs)


DEFAULT_QUERY_PROMPT_EN = PromptTemplate(
    """You are an AI language model assistant. Your task is 
        to generate 3 different versions of the given user 
        question to retrieve relevant documents from a vector database. 
        Your goal is to help the user overcome some of the limitations 
        of distance-based similarity search. Provide these alternative 
        questions in english.Please number the answers starting from 1 
        and separated by newlines. Original question: {question}"""
)

DEFAULT_QUERY_PROMPT_CH = PromptTemplate(
    """你是一个人工智能语言模型助理。您的任务是根据用户的原始问题，从矢量数据库中基于
    距离的相似性检索出与原问题相关的3个问题。你的目标是通过生成的多个角度的提问来帮助
    用户克服原问题的限制。请从1开始编号且用中文回答，每个回答用换行符分隔开。原始问题：{question}"""
)


class OutputParser(ABC):
    @abstractmethod
    def parse(self, output: str) -> List[str]:
        """Parse multi query response, convert to line list.
        Args:
            output: String to find relevant documents for
        Returns:
            List of line
        """


class DefaultOutputParser(OutputParser):
    @staticmethod
    def _is_starting_with_number(query: str):
        return bool(re.match(r'\d.*', query))

    def parse(self, output: str) -> List[str]:
        lines = []
        for line in output.splitlines():
            if self._is_starting_with_number(line.strip()):
                lines.append(line)
        return lines


class MultiQueryRetriever(Retriever):
    def __init__(self, llm: Text2TextLLM,
                 prompt: PromptTemplate = DEFAULT_QUERY_PROMPT_CH,
                 parser: OutputParser = DefaultOutputParser(),
                 max_tokens: int = 512,
                 **data: Any):
        super().__init__(**data)
        self._llm = llm
        self._prompt = prompt
        self._parser = parser
        self._max_tokens = max_tokens

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = []

        llm_query = self._prompt.format(question=query)
        llm_response = self._llm.chat(query=llm_query, history=[], role="user", max_tokens=self._max_tokens)
        for sub_query in self._parser.parse(output=str(llm_response)):
            logger.success(f"sub_query {sub_query}")
            doc = super()._get_relevant_documents(sub_query)
            docs.extend(doc)

        docs = [doc for i, doc in enumerate(docs) if doc not in docs[:i]]
        return sorted(docs, key=lambda x: len(x.page_content))
