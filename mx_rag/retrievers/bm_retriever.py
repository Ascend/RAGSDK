# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import List, Callable
from langchain_core.pydantic_v1 import validator, Field

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from mx_rag.utils.common import MAX_TOP_K

from mx_rag.llm import Text2TextLLM

_KEY_WORD_TEMPLATE_ZH = PromptTemplate(
    input_variables=["user_request"],
    template="""根据问题提取关键词，不超过10个。关键词尽量切分为动词、名词、或形容词等单独的词，
不要长词组（目的是更好的匹配检索到语义相关但表述不同的相关资料）。请根据给定参考资料提前关键词，关键词之间使用逗号分隔，比如{{关键词1, 关键词2}}
Question: CANN如何安装？
Keywords: CANN, 安装, install

Question: MindStudio 容器镜像怎么制作
Keywords: MindStudio, 容器镜像, Docker build

Question: {user_request}
Keywords:
""")


def _default_preprocessing_func(text: str) -> List[str]:
    return text.split(",")


class BMRetriever(BaseRetriever):
    docs: List[Document]
    llm: Text2TextLLM
    k: int = Field(default=1, ge=1, le=MAX_TOP_K)
    max_tokens = 512
    temperature = 0.5
    top_p = 0.95
    prompt: PromptTemplate = _KEY_WORD_TEMPLATE_ZH

    class Config:
        arbitrary_types_allowed = True

    @property
    def preprocess_func(self) -> Callable[[str], List[str]]:
        return _default_preprocessing_func

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        res = self.llm.chat(self.prompt.format(user_request=query), max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            top_p=self.top_p)

        if not res.strip():
            raise ValueError("generate keywords failed")

        if not self.docs:
            return []

        retriever = BM25Retriever.from_documents(documents=self.docs, bm25_params=None,
                                                 preprocess_func=self.preprocess_func, k=self.k)

        return retriever.invoke(res)
