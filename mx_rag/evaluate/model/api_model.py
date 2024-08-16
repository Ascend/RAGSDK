# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import typing as t
from typing import List

from langchain.schema import LLMResult
from langchain.schema import Generation
from ragas.llms import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.prompt import PromptValue
from ragas.run_config import RunConfig

from mx_rag.llm.text2text import Text2TextLLM
from mx_rag.embedding.service.tei_embedding import TEIEmbedding


class APILLM(BaseRagasLLM):

    def __init__(self,
                 llm_url: str,
                 model_name: str,
                 cert_file: str = "",
                 crl_file: str = "",
                 use_http: bool = False):
        self.url = llm_url
        self.llm = Text2TextLLM(base_url=self.url,
                                model_name=model_name,
                                timeout=10 * 60,
                                cert_file=cert_file,
                                crl_file=crl_file,
                                use_http=use_http)

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

    def __init__(self, embed_url: str, max_length=512, batch_size=256, use_http: bool = False):
        self.url = embed_url
        self.embed = TEIEmbedding(embed_url, use_http=use_http)
        self.max_length = max_length
        self.batch_size = batch_size
        self.set_run_config(RunConfig())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed.embed_documents(texts, self.batch_size)

    def embed_query(self, text: str) -> List[float]:
        result = self.embed.embed_documents([text], self.batch_size)
        return result[0] if result else []
