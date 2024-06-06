# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import re
from abc import abstractmethod, ABC
from typing import List, Any

from loguru import logger

from mx_rag.document.loader.docx_loader import Doc
from mx_rag.llm import MindieLLM
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


DEFAULT_QUERY_PROMPT = PromptTemplate(
    """You are an AI language model assistant. Your task is 
        to generate 3 different versions of the given user 
        question to retrieve relevant documents from a vector database. 
        By generating multiple perspectives on the user question, 
        your goal is to help the user overcome some of the limitations 
        of distance-based similarity search. Provide these alternative 
        questions separated by newlines. Original question: {question}"""
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
    def parse(self, output: str) -> List[str]:
        lines = []
        for line in output.splitlines():
            if line.strip() == "":
                continue
            lines.append(line)
        return lines


class MultiQueryRetriever(Retriever):
    def __init__(self, llm: MindieLLM,
                 prompt: PromptTemplate = DEFAULT_QUERY_PROMPT,
                 parser: OutputParser = DefaultOutputParser(),
                 **data: Any):
        super().__init__(**data)
        self._llm = llm
        self._prompt = prompt
        self._parser = parser

    def _get_relevant_documents(self, query: str) -> List[Doc]:
        docs = []

        llm_query = self._prompt.format(question=query)
        llm_response = self._llm.chat(query=llm_query, history=[], max_tokens=2048)
        for sub_query in self._parser.parse(output=str(llm_response)):
            doc = super()._get_relevant_documents(sub_query)
            docs.extend(doc)

        return sorted(list(set(docs)))
