# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import typing as t
import json
import asyncio
from typing import List
import requests
from ragas.llms import BaseRagasLLM
from langchain.schema import LLMResult
from langchain.schema import Generation
from langchain.callbacks.base import Callbacks
from langchain.schema.embeddings import Embeddings
from ragas.llms.prompt import PromptValue
from loguru import logger
from mx_rag.utils import RequestUtils


class APILLM(BaseRagasLLM):

    def __init__(self, llm_url: str):
        self.url = llm_url
        self.client = RequestUtils()
        self.headers = {
            'Content-Type': 'application/json'
        }

    @property
    def llm(self):
        return self.base_llm

    def get_llm_result(self, prompt):
        generations = []
        llm_output = {}
        token_total = 0
        content = prompt.to_string()
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        })
        resp = self.client.post(self.url, payload, headers=self.headers)
        if not resp.success:
            logger.error("llm api request failed")
            return ""

        data = json.loads(resp.text)['result']
        generations.append([Generation(text=data)])
        token_total += len(data)
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


class APIEmbedding(Embeddings):

    def __init__(self, embed_url: str, max_length=512, batch_size=256):
        self.url = embed_url
        self.headers = {
            'Content-Type': 'application/json'
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = json.dumps({
            "input": texts
        })
        resp = self.client.post(self.url, payload, headers=self.headers)
        if not resp.success:
            logger.error("embedding url request failed")
            return [[]]
        result_list = []
        for i in range(len(texts)):
            result = json.loads(resp.text)['data'][i]['embedding']
            result_list.append(result)
        return result_list

    def embed_query(self, text: str) -> List[float]:
        payload = json.dumps({
            "input": [text]
        })
        resp = self.client.post(self.url, payload, headers=self.headers)
        if not resp.success:
            logger.error("embedding url request failed")
            return [[]]
        result = json.loads(resp.text)['data'][0]['embedding']
        return result