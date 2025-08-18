# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import copy
from typing import Union, Iterator, List, Dict

from langchain_core.documents import Document
from loguru import logger

from langchain_core.retrievers import BaseRetriever

from mx_rag.utils.common import validate_params, BOOL_TYPE_CHECK_TIP, TEXT_MAX_LEN
from mx_rag.llm.llm_parameter import LLMParameterConfig
from mx_rag.chain import Chain
from mx_rag.llm import Text2TextLLM
from mx_rag.reranker.reranker import Reranker
from mx_rag.utils.common import MAX_PROMPT_LENGTH

TEXT_INFER_PRMPT = '''
You are a helpful question-answering assistant. Your task is to generate a interleaved text and image response based on provided questions and quotes. Here‘s how to refine your process:

1. **Evidence Selection**:
   - From both text and image quotes, pinpoint those really relevant for answering the question. Focus on significance and direct relevance.
   - Each image quote is the description of the image.

2. **Answer Construction**:
   - Use Markdown to embed text and images in your response, avoid using obvious headings or divisions; ensure the response flows naturally and cohesively.
   - Conclude with a direct and concise answer to the question in a simple and clear sentence.

3. **Quote Citation**:
   - Cite text by adding [index]; for example, quote from the first text should be [1].
   - Cite images using the format `![{conclusion}](image index)`; for the first image, use `![{conclusion}](image1)`;The {conclusion} should be a concise one-sentence summary of the image’s content.
   - Ensure the cite of the image must strict follow `![{conclusion}](image index)`, do not simply stating "See image1", "image1 shows" ,"[image1]" or "image1".
   - Each image or text can only be quoted once.

- Do not cite irrelevant quotes.
- Compose a detailed and articulate interleaved answer to the question.
- Ensure that your answer is logical, informative, and directly ties back to the evidence provided by the quotes.
- Interleaved answer must contain both text and image response.
- Answer in chinese.
'''


class Multi2MultiChain(Chain):
    document_separator: str = "\n\n"

    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM), message="param must be instance of Text2TextLLM"),
        retriever=dict(validator=lambda x: isinstance(x, BaseRetriever),
                       message="param must be instance of BaseRetriever"),
        reranker=dict(validator=lambda x: isinstance(x, Reranker) or x is None,
                      message="param must be None or instance of Reranker"),
        prompt=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= MAX_PROMPT_LENGTH,
                    message=f"param must be str and length range [1, {MAX_PROMPT_LENGTH}]"),
        source=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
    )
    def __init__(self, llm: Text2TextLLM,
                 retriever: BaseRetriever,
                 reranker: Reranker = None,
                 prompt: str = TEXT_INFER_PRMPT,
                 source: bool = True):
        super().__init__()
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm
        self._prompt = prompt
        self._source = source
        self._role: str = "user"

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= TEXT_MAX_LEN,
                  message=f"param must be a str and its length meets (0, {TEXT_MAX_LEN}]"),
        llm_config=dict(validator=lambda x: isinstance(x, LLMParameterConfig),
                        message="llm_config must be instance of LLMParameterConfig")
    )
    def query(self, text: str, llm_config: LLMParameterConfig = LLMParameterConfig(temperature=0.5, top_p=0.95),
              *args, **kwargs) \
            -> Dict:
        return self._query(text, llm_config)

    def _merge_query_prompt(self, query: str, docs: List[Document]):
        # 拆分知识片段分为原始文本和图片多粒度信息文本
        text_docs = [doc for doc in docs if doc.metadata.get("type", "") == "text"]
        img_docs = [doc for doc in docs if doc.metadata.get("type", "") == "image"]

        # 2. Add text quotes
        user_message = "Text Quotes are:"
        for i, doc in enumerate(text_docs):
            user_message += f"\n[{i + 1}] {doc.page_content}"
        if len(img_docs) > 0:
            # 3. Add image quotes vlm-text or ocr-text
            user_message += "\nImage Quotes are:"
            for i, doc in enumerate(img_docs):
                user_message += f"\nimage{i + 1} is described as: {doc.page_content}"

        user_message += "\n\n"

        # 4. add user question
        user_message += f"The user question is: {query}"

        return user_message

    def _query(self,
               question: str,
               llm_config: LLMParameterConfig) -> Union[Dict, Iterator[Dict]]:

        q_docs = self._retriever.invoke(question)

        if self._reranker is not None and len(q_docs) > 0:
            scores = self._reranker.rerank(question, [doc.page_content for doc in q_docs])
            q_docs = self._reranker.rerank_top_k(q_docs, scores)

        q_with_prompt = self._merge_query_prompt(question, copy.deepcopy(q_docs))

        return self._do_query(q_with_prompt, llm_config, question=question, q_docs=q_docs)

    def _do_query(self, q_with_prompt, llm_config: LLMParameterConfig, question: str, q_docs: List[Document]) \
            -> Dict:
        logger.info("invoke normal query")
        resp = {"query": question, "result": ""}
        if self._source:
            resp['source_documents'] = [{'metadata': x.metadata, 'page_content': x.page_content} for x in q_docs]

        sys_messages = [{"role": "system", "content": self._prompt}]

        llm_response = self._llm.chat(query=q_with_prompt, sys_messages=sys_messages, role=self._role, llm_config=llm_config)
        resp['result'] = llm_response
        return resp

