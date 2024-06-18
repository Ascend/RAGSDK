# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import typing as t
from typing import List

from langchain.schema import LLMResult
from langchain.schema import Generation
from ragas.llms import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.prompt import PromptValue

from mx_rag.llm.text2text import Text2TextLLM
from mx_rag.embedding.service.tei_embedding import TEIEmbedding


class APILLM(BaseRagasLLM):

    def __init__(self, llm_url: str, model_name: str):
        self.url = llm_url
        self.llm = Text2TextLLM(self.url, model_name, timeout=60 * 10)

    def generate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 1e-8,
            stop: t.Optional[t.List[str]] = None,
            callbacks=None,
    ):
        generations = []
        llm_output = {}

        generate_texts = self.llm.chat(prompt.to_string(), [], temperature=temperature, max_tokens=1024)

        generations.append([Generation(text=generate_texts)])
        llm_output['token_total'] = len(generate_texts)
        result = LLMResult(generations=generations, llm_output=llm_output)
        return result

    async def agenerate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 1e-8,
            stop: t.Optional[t.List[str]] = None,
            callbacks=None,
    ) -> LLMResult:
        return self.generate_text(prompt, n, temperature, stop, callbacks)


class APIEmbedding(BaseRagasEmbeddings):

    def __init__(self, embed_url: str, max_length=512, batch_size=256):
        self.url = embed_url
        self.embed = TEIEmbedding(embed_url)
        self.max_length = max_length
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self.embed.embed_texts(texts, self.batch_size)
        result = result.tolist()
        return result if result else [[]]

    def embed_query(self, text: str) -> List[float]:
        result = self.embed.embed_texts([text], self.batch_size)
        result = result.tolist()
        return result[0] if result else []
