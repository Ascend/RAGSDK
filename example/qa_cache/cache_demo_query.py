# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

# pylint: disable=duplicate-code

import argparse
import os
import time
import traceback

from loguru import logger
from pymilvus import MilvusClient

from mx_rag.cache import CacheConfig, SimilarityCacheConfig, MxRAGCache, CacheChainChat
from mx_rag.chain import SingleText2TextChain
from mx_rag.embedding.service import TEIEmbedding
from mx_rag.llm import Text2TextLLM
from mx_rag.retrievers import Retriever
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.storage.vectorstore import MilvusDB
from mx_rag.utils import ClientParam


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_default_metavar_for_optional(self, action):
        return action.type.__name__

    def _get_default_metavar_for_positional(self, action):
        return action.type.__name__


def rag_demo_query():
    parse = argparse.ArgumentParser(formatter_class=CustomFormatter)
    parse.add_argument(
        "--embedding_url",
        type=str,
        help="向量化服务地址，例如: http://127.0.0.1:8080/v1/embeddings",
    )
    parse.add_argument(
        "--llm_url",
        type=str,
        help="大模型服务地址，例如: http://127.0.0.1:1025/v1/chat/completions",
    )
    parse.add_argument(
        "--model_name",
        type=str,
        default="Llama3-8B-Chinese-Chat",
        help="大模型服务名：例如: Llama3-8B-Chinese-Chat",
    )
    parse.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="相似性得分的阈值，大于阈值认为检索的信息与问题越相关,取值范围[0,1]",
    )
    parse.add_argument(
        "--tei_reranker",
        type=bool,
        default=False,
        help="是否使用TEI服务化的reranker模型",
    )
    parse.add_argument(
        "--reranker_url",
        type=str,
        help="排序模型服务地址，例如http://127.0.0.1:8080/v1/rerank",
    )
    parse.add_argument("--top_k", type=int, default=3, help="相似度排序top值")
    parse.add_argument("--query", type=str, help="用户查询问题")
    args = parse.parse_args()

    cache_save_path = os.path.split(os.path.realpath(__file__))[0]
    # 加载embedding模型，请根据模型具体路径适配
    client_param = ClientParam(use_http=True, timeout=600)
    emb = TEIEmbedding(url=args.embedding_url, client_param=client_param)
    embedding_dim = len(emb.embed_query("你好"))
    llm = Text2TextLLM(base_url=args.llm_url, model_name=args.model_name, client_param=client_param)

    client = MilvusClient("./vector.db")

    vector_store = MilvusDB.create(client=client, x_dim=embedding_dim)
    # 初始化文档chunk关系数据库
    chunk_store = SQLiteDocstore(db_path="./chunk.db")
    # 初始化知识管理关系数据库

    # memory cache缓存作为L1缓存
    cache_config = CacheConfig(cache_size=100, data_save_folder=cache_save_path)
    # similarity cache缓存作为L2缓存
    similarity_config = SimilarityCacheConfig(
        vector_config={
            "vector_type": "milvus_db",
            "x_dim": embedding_dim,
            "client": client,
        },
        cache_config="sqlite",
        emb_config={
            "embedding_type": "tei_embedding",
            "url": args.embedding_url,
            "client_param": client_param,
        },
        similarity_config={
            "similarity_type": "tei_reranker",
            "url": args.reranker_url,
            "k": args.top_k,
            "client_param": client_param,
        },
        retrieval_top_k=5,
        cache_size=1000,
        similarity_threshold=0.8,
        data_save_folder=cache_save_path,
    )

    # 构造memory cache实例
    memory_cache = MxRAGCache("memory_cache", cache_config)
    # 构造similarity cache实例
    similarity_cache = MxRAGCache("similarity_cache", similarity_config)
    # memory_cache和similarity_cache串联形成多级缓存入口是memory cache
    memory_cache.join(similarity_cache)
    try:
        # 初始化Retriever检索器
        text_retriever = Retriever(
            vector_store=vector_store,
            document_store=chunk_store,
            embed_func=emb.embed_documents,
            k=args.top_k,
            score_threshold=args.score_threshold,
        )

        # 构造cache_chain, 缓存memory cache作为入口
        cache_chain = CacheChainChat(chain=SingleText2TextChain(llm, text_retriever), cache=memory_cache)
        # 提问和网页相关的问题，如果与已生成的QA近似，则会命中返回
        now_time = time.time()
        logger.info(cache_chain.query(args.query))
        logger.info(f"耗时:{time.time() - now_time}s")

    except Exception:
        stack_trace = traceback.format_exc()
        logger.error(stack_trace)


if __name__ == "__main__":
    rag_demo_query()
