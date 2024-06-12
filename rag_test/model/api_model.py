import typing as t
import json
import requests
from typing import List
from ragas.llms import BaseRagasLLM
from langchain.schema import LLMResult
from langchain.schema import Generation
from langchain.callbacks.base import Callbacks
from langchain.schema.embeddings import Embeddings
from ragas.llms.prompt import PromptValue
from transformers import AutoTokenizer, pipeline

class APILLM(BaseRagasLLM):

    def __init__(self, llm_url):
        self.url = llm_url
        self.headers = {
            'Content-Type': 'application/json'
        }

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
        response = requests.request("POST", self.url, headers=self.headers, data=payload)
        data = json.loads(response.text)['result']
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
            callbacks: Callbacks = [],
    ):
        result = self.get_llm_result(prompt)
        return result

class APIEmbedding(Embeddings):
    def __init__(self, embed_url):
        self.url = embed_url
        self.headers = {
            'Content-Type': 'application/json'
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = json.dumps({
            "input": texts
        })
        response = requests.request("POST", self.url, headers=self.headers, data=payload)
        embeddings_list=[]
        for i in range(len(texts)):
            embeddings=json.loads(response.text)['data'][i]['embedding']
            embeddings_list.append(embeddings)
        return embeddings_list

    def embed_query(self, text: str) -> List[float]:
        payload = json.dumps({
            "input": [text]
        })
        response = requests.request("POST", self.url, headers=self.headers, data=payload)
        embeddings=json.loads(response.text)['data'][0]['embedding']
        return embeddings