# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import Union, Dict, Iterator
from multiprocessing import Process, Value, Lock, Queue

from mx_rag.chain.single_text_to_text import SingleText2TextChain


class ParallelText2TextChain(SingleText2TextChain):
    FIRST_RAG_PROMPT = (
        "根据已知信息，简洁和专业的来回答用户的问题。如果无法从中已知信息中得到答案，请根据自身经验做出的的回答 用户问题:"
    )

    NEXT_RAG_PROMPT = (
        "下面是已知信息:"
    )

    def __init__(self, prompt: str = FIRST_RAG_PROMPT, **kwargs):
        super().__init__(prompt=prompt, **kwargs)
        self.prefill_done = Value('i', 0)
        self.prefill_queue = Queue()
        self.lock = Lock()

    def query(self, text: str, *args, **kwargs) -> Union[Dict, Iterator[Dict]]:
        """
        推理和检索并行查询 query
        首先开启prefill 推理检测进程，之后进行检索过程，如果检索完成，prefill未完成则此次回答包含检索内容
        如果prefill完成，检索未完成则此次回答不包含检索内容

        Args:
            text: 用户查询问题
            *args: 参考SingleText2TextChain
            **kwargs: 参考SingleText2TextChain

        Returns:
            用户答案
        """
        self._query_str = text

        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.5)
        top_p = kwargs.get("top_p", 0.95)

        # 启动prefill检测进程
        prefill_process = Process(target=self._prefill_process, args=(text, max_tokens, temperature, top_p))
        prefill_process.start()

        # 执行检索
        self._retrieve_process(text)

        # 检测prefill是否完成
        with self.lock:
            prefill_is_done = True if self.prefill_done.value == 1 else False

        # 如果prefill 已经完成则使用prefill结果
        if prefill_is_done:
            answer = self.prefill_queue.get(block=True, timeout=60)
            answer = answer[0]
        # 否则 走正常推理流程
        else:
            question = self._prompt + text + "\n" + self.NEXT_RAG_PROMPT
            for doc in self._docs:
                question = question + doc.page_content

            answer = self._do_query(question, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

        prefill_process.join()
        self.prefill_done.value = 0
        return answer

    def _prefill_process(self, text, max_tokens, temperature, top_p):
        """
        执行prefill 检测，如果prefill已经完成就把 prefill_done标志位置为1，并继续返回流式推理结果
        Args:
            text: 用户问题

        Returns:
            流式推理结果
        """
        question = self._prompt + text
        answer_interator = self._do_stream_query(question, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        result = ""

        for ans in answer_interator:
            result = ans
            if self.prefill_done.value == 0:
                with self.lock:
                    self.prefill_done.value = 1

        self.prefill_queue.put([result])

    def _retrieve_process(self, text: str):
        """
        执行检索，通过检索和reranker
        Args:
            text: 用户问题

        Returns:
            流式推理结果
        """
        if self._retriever is not None:
            self._docs = self._retriever.get_relevant_documents(text)

        if self._reranker is not None:
            scores = self._reranker.rerank(text, [doc.page_content for doc in self._docs])
            self._docs = self._reranker.rerank_top_k(self._docs, scores)
