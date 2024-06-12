import typing as t
from typing import List
from ragas.llms import BaseRagasLLM
from langchain.schema import LLMResult
from langchain.schema import Generation
from langchain.callbacks.base import Callbacks
from langchain.schema.embeddings import Embeddings
from FlagEmbedding import FlagModel
from ragas.llms.prompt import PromptValue
from transformers import AutoTokenizer, pipeline

class LocalLLM(BaseRagasLLM):

    def __init__(self, llm_path):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.qa_pipeline = pipeline(
            "text-generation",
            model=llm_path,
            device=torch.device('npu'),
        )

    def generate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 0.7,
            stop: t.Optional[t.List[str]] = None,
            callbacks: Callbacks = [],
    ):
        generations = []
        llm_output = {}
        token_total = 0
        messages=[
            {"role": "user", "content": prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.qa_pipeline(
                prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95
            )
        data = outputs[0]["generated_text"][len(prompt):]
        generations.append([Generation(text=data)])
        token_total += len(data)
        llm_output['token_total'] = token_total
        return LLMResult(generations=generations, llm_output=llm_output)

class LocalEmbedding(Embeddings):
    def __init__(self, path, max_length=512, batch_size=256):
        self.model = FlagModel(path)
        self.max_length = max_length
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode_corpus(texts, self.batch_size, self.max_length).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode_queries(text, self.batch_size, self.max_length).tolist()