# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import typing as t
import asyncio
from typing import List
from ragas.llms import BaseRagasLLM
from langchain.schema import LLMResult
from langchain.schema import Generation
from langchain.callbacks.base import Callbacks
from langchain.schema.embeddings import Embeddings
from FlagEmbedding import FlagModel
from transformers import AutoModel, AutoTokenizer
from ragas.llms.prompt import PromptValue


class LocalLLM(BaseRagasLLM):

    def __init__(self, llm_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        self.base_llm = AutoModel.from_pretrained(llm_path, trust_remote_code=True).npu()
        self.base_llm = self.base_llm.eval()

    @property
    def llm(self):
        return self.base_llm

    def get_llm_result(self, prompt):
        generations = []
        llm_output = {}
        token_total = 0
        content = prompt.to_string()
        text, history = self.base_llm.chat(self.tokenizer, content, history=[])
        generations.append([Generation(text=text)])
        token_total += len(text)
        llm_output['token_total'] = token_total
        return LLMResult(generations=generations, llm_output=llm_output)

    def generate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 1e-8,
            stop: t.Optional[t.List[str]] = None,
            callbacks=None,
    ):
        result = self.get_llm_result(prompt)
        return result

    async def agenerate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 1e-8,
            stop: t.Optional[t.List[str]] = None,
            callbacks=None,
    ) -> LLMResult:
        pass


class LocalEmbedding(Embeddings):

    def __init__(self, path: str, max_length=512, batch_size=256):
        self.model = FlagModel(path, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
        self.max_length = max_length
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode_corpus(texts, self.batch_size, self.max_length).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode_queries(text, self.batch_size, self.max_length).tolist()