# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import List, Callable

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import field_validator, Field, ConfigDict

from mx_rag.llm import Text2TextLLM
from mx_rag.llm.llm_parameter import LLMParameterConfig
from mx_rag.utils.common import MAX_TOP_K, MAX_PROMPT_LENGTH, TEXT_MAX_LEN, validate_params

_KEY_WORD_TEMPLATE_ZH = PromptTemplate(
    input_variables=["question"],
    template="""根据问题提取关键词，不超过10个。关键词尽量切分为动词、名词、或形容词等单独的词，
不要长词组（目的是更好的匹配检索到语义相关但表述不同的相关资料）。请根据给定参考资料提前关键词，关键词之间使用逗号分隔，比如{{关键词1, 关键词2}}
Question: CANN如何安装？
Keywords: CANN, 安装, install

Question: MindStudio 容器镜像怎么制作
Keywords: MindStudio, 容器镜像, Docker build

Question: {question}
Keywords:
""")


def _default_preprocessing_func(text: str) -> List[str]:
    return text.split(",")


class BMRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    docs: List[Document]
    llm: Text2TextLLM
    k: int = Field(default=1, ge=1, le=MAX_TOP_K)
    llm_config: LLMParameterConfig = LLMParameterConfig(temperature=0.5, top_p=0.95)
    preprocess_func: Callable[[str], List[str]] = None
    prompt: PromptTemplate = _KEY_WORD_TEMPLATE_ZH

    @field_validator('prompt')
    @classmethod
    def _validate_prompt(cls, prompt):
        if set(prompt.input_variables) != {"question"}:
            raise ValueError('prompt.input_variables must include exactly "question".')
        if not (0 < len(prompt.template) <= MAX_PROMPT_LENGTH):
            raise ValueError(f'prompt.template length must be between 1 and {MAX_PROMPT_LENGTH}.')
        return prompt

    @validate_params(
        query=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
                   message=f"query must be a str and length range (0, {TEXT_MAX_LEN}]")
    )
    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        res = self.llm.chat(self.prompt.format(question=query), llm_config=self.llm_config)

        if not res.strip():
            raise ValueError("generate keywords failed")

        if not self.docs:
            return []

        preprocess_function = _default_preprocessing_func if self.preprocess_func is None else self.preprocess_func

        retriever = BM25Retriever.from_documents(documents=self.docs, bm25_params=None,
                                                 preprocess_func=preprocess_function, k=self.k)

        return retriever.invoke(res)
