# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Union, Dict, Iterator

from loguru import logger

from mx_rag.utils.common import validate_params
from mx_rag.llm import Text2TextLLM
from mx_rag.chain import SingleText2TextChain
from mx_rag.llm import Text2TextLLM

if TYPE_CHECKING:
    from mx_rag.retrievers import TreeRetriever


class TreeText2TextChain(SingleText2TextChain):

    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM)),
    )
    def __init__(self, llm, retriever: TreeRetriever = None):
        super().__init__(llm, retriever)
        self._llm = llm
        self.tree_retriever = retriever
        self._source = False

    def query(self, text: str, *args, **kwargs) -> Union[Dict, Iterator[Dict]]:
        self._history = []
        return self._answer_question(text, *args, **kwargs)

    def set_tree_retriever(self, tree_retriever: TreeRetriever):
        self.tree_retriever = tree_retriever

    def summarize(self, text: str,
                  max_tokens: int = 100,
                  temperature: float = 0.5,
                  top_p: float = 0.95) -> Union[Dict, Iterator[Dict]]:
        # 不带历史内容
        self._history = []
        self._query_str = text
        question = f"为以下内容生成摘要，包含尽可能多的关键细节，请用中文回答: \n\n{text}:"
        return self._do_query(question, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def _answer_question(
            self,
            text,
            temperature: float = 0.5,
            top_p: float = 0.95,
            max_tokens: int = 512,
            stream: bool = False
    ):

        context = self._retrieve(text)
        final_question = (f"参考信息: {context} 我的问题或指令：{text} \n请根据上述参考信息回答我的问题或回复我的指令。"
                          f"前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你的回答提供依据。"
                          f"回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复, 你的回复: ")
        self._query_str = final_question
        logger.info(f"the fianl question to llm : {final_question}")
        if not stream:
            return self._do_query(final_question, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        return self._do_stream_query(final_question, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def _retrieve(
            self,
            question
    ):
        if self.tree_retriever is None:
            raise ValueError("The TreeRetriever instance has not been initialized. Call 'add_documents' first.")

        return self.tree_retriever.retrieve(
            question
        )
