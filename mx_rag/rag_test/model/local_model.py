# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import typing as t
from typing import List

from langchain.schema import LLMResult
from langchain.schema import Generation
from transformers import AutoModel, AutoTokenizer
from ragas.llms import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.prompt import PromptValue

from mx_rag.embedding.local.text_embedding import TextEmbedding


class LocalLLM(BaseRagasLLM):

    def __init__(self, llm_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        self.base_llm = AutoModel.from_pretrained(llm_path, trust_remote_code=True).npu()
        self.base_llm = self.base_llm.eval()

    @property
    def llm(self):
        return self.base_llm

    def generate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 1e-8,
            stop: t.Optional[t.List[str]] = None,
            callbacks=None,
    ):
        result = self._get_llm_result(prompt)
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

    def _get_llm_result(self, prompt):
        generations = []
        llm_output = {}
        token_total = 0
        content = prompt.to_string()
        text, history = self.base_llm.chat(self.tokenizer, content, history=[])
        generations.append([Generation(text=text)])
        token_total += len(text)
        llm_output['token_total'] = token_total
        return LLMResult(generations=generations, llm_output=llm_output)


class LocalEmbedding(BaseRagasEmbeddings):

    def __init__(self, path: str, max_length=512, batch_size=256):
        self.model = TextEmbedding(path)
        self.max_length = max_length
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_texts(texts, self.batch_size, self.max_length).tolist()

    def embed_query(self, text: str) -> List[float]:
        result = self.model.embed_texts([text], self.batch_size, self.max_length).tolist()
        return result[0] if result else []
