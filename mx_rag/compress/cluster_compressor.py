# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import re
from typing import List, Callable
from langchain_text_splitters.base import TextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
import torch
import numpy as np
import torch.nn.functional as F

from mx_rag.compress import PromptCompressor
from mx_rag.utils.common import validate_params, STR_TYPE_CHECK_TIP, MAX_PAGE_CONTENT, MAX_DEVICE_ID, \
    MAX_QUERY_LENGTH


class ClusterCompressor(PromptCompressor):
    @validate_params(
        dev_id=dict(validator=lambda x: isinstance(x, int) and 0 <= x <= MAX_DEVICE_ID,
                    message="param must be int and value range [0, 63]"),
        embed=dict(validator=lambda x: isinstance(x, Embeddings),
                   message="param must be instance of LangChain's Embeddings"),
        cluster_func=dict(validator=lambda x: isinstance(x, Callable),
                          message="param must be Callable[[List[List[float]]], List[int]] function"),
        splitter=dict(validator=lambda x: isinstance(x, TextSplitter) or x is None,
                      message="param must be instance of LangChain's TextSplitter or None"),
    )
    def __init__(self,
                 cluster_func: Callable[[List[List[float]]], List[int]],
                 embed: Embeddings,
                 splitter: TextSplitter = None,
                 dev_id: int = 0,
                 ):
        self.embed = embed
        self.cluster_func = cluster_func
        self.splitter = splitter
        self.dev_id = dev_id

    @staticmethod
    def _assemble_result(sentences, labels, similarity, target_rate):
        # 根据压缩率，每个社区删除对应的比例，相似性差的先删
        reserved_sentences = []
        community = {}
        for index, label in enumerate(labels):
            if label not in community:
                community[label] = [index]
            else:
                community[label].append(index)
        for _, value in community.items():
            similarity_temp = [similarity[i] for i in value]
            sorted_sentences = np.argsort(similarity_temp)
            reserved_index = sorted_sentences[int(len(value) * target_rate):]
            for left_index in reserved_index:
                reserved_sentences.append(value[left_index])

        reserved_sentences = sorted(reserved_sentences)
        compress_context = ''.join([sentences[i] for i in reserved_sentences])
        return compress_context

    @validate_params(
        context=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= MAX_PAGE_CONTENT,
                     message=f"param must be str, and length range [1, {MAX_PAGE_CONTENT}]"),
        question=dict(validator=lambda x: isinstance(x, str) and 1 <= len(x) <= MAX_QUERY_LENGTH,
                      message=STR_TYPE_CHECK_TIP + f", and length range [1, {MAX_QUERY_LENGTH}]"),
        target_rate=dict(validator=lambda x: isinstance(x, float) and 0 < x < 1,
                         message=f"param must be float and value range (0, 1)"),
    )
    def compress_texts(self,
                       context: str,
                       question: str,
                       target_rate: float = 0.6,
                       ):
        if self.splitter is None:
            sentence_splitter = RecursiveCharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=0,
                separators=["。", "！", "？", "\n", "，", "；", " ", ""],  # 中文分隔符列表
            )
            self.splitter = sentence_splitter
        # 文本切分
        sentences = self.splitter.split_text(text=context)

        if len(sentences) < 2:
            return context
        # 文本embedding
        sentences_with_question = sentences + [question]
        sentences_embedding_with_question = self.embed.embed_documents(sentences_with_question)
        sentences_embedding = sentences_embedding_with_question[:-1]
        question_embedding = sentences_embedding_with_question[-1]
        # 计算余弦相似度
        similarity = np.array(self._get_similarity(sentences_embedding, question_embedding).to('cpu'))
        # 社区划分
        label = self.cluster_func(sentences_embedding)
        # 压缩文本
        compress_context = self._assemble_result(sentences, label, similarity, target_rate)

        return compress_context

    def _get_similarity(self, sentences_embedding, question_embedding):
        # 计算句子和问题的相似度
        sentences_embedding = np.array(sentences_embedding, dtype=np.float32)
        question_embedding = np.array(question_embedding, dtype=np.float32)

        c1 = torch.from_numpy(sentences_embedding).to(f'npu:{self.dev_id}')
        c2 = F.normalize(c1, p=2, dim=-1)

        q1 = torch.from_numpy(question_embedding).to(f'npu:{self.dev_id}')
        q2 = F.normalize(q1, p=2, dim=-1)

        sims_with_query = q2.squeeze() @ c2.T  # 余弦相似度
        return sims_with_query
