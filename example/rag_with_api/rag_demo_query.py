# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

# pylint: disable=duplicate-code

import argparse
import threading
import traceback

from loguru import logger
from pymilvus import MilvusClient

from mx_rag.chain import SingleText2TextChain
from mx_rag.embedding.service import TEIEmbedding
from mx_rag.llm import Text2TextLLM
from mx_rag.retrievers import Retriever
from mx_rag.storage.document_store import MilvusDocstore
from mx_rag.storage.vectorstore import MilvusDB
from mx_rag.utils import ClientParam


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_default_metavar_for_optional(self, action):
        return action.type.__name__

    def _get_default_metavar_for_positional(self, action):
        return action.type.__name__


class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=None, kwargs=None, *, daemon=None):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        self.result = None

        def function():
            self.result = target(*args, **kwargs)

        super().__init__(group=group, target=function, name=name, daemon=daemon)


def rag_demo_query():
    parse = argparse.ArgumentParser(formatter_class=CustomFormatter)
    parse.add_argument(
        "--embedding_url",
        type=str,
        default="http://127.0.0.1:8080/embed",
        help="使用TEI服务化的embedding模型url地址",
    )
    parse.add_argument(
        "--llm_url",
        type=str,
        default="http://127.0.0.1:1025/v1/chat/completions",
        help="大模型url地址",
    )
    parse.add_argument("--model_name", type=str, default="Llama3-8B-Chinese-Chat", help="大模型名称")
    parse.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="相似性得分的阈值，大于阈值认为检索的信息与问题越相关,取值范围[0,1]",
    )
    parse.add_argument("--query", type=str, action="append", help="用户问题")
    parse.add_argument("--num_threads", type=int, default=2, help="可以根据实际情况调整线程数量")

    args = parse.parse_args().__dict__
    embedding_url: str = args.pop("embedding_url")
    llm_url: str = args.pop("llm_url")
    model_name: str = args.pop("model_name")
    score_threshold: float = args.pop("score_threshold")
    query: list[str] = args.pop("query")
    num_threads: int = args.pop("num_threads")

    try:
        # 加载embedding模型，请根据模型具体路径适配
        emb = TEIEmbedding(url=embedding_url, client_param=ClientParam(use_http=True))
        embedding_dim = len(emb.embed_documents(["test"])[0])
        # 初始化向量数据库
        client = MilvusClient("./milvus.db")
        vector_store = MilvusDB.create(client=client, x_dim=embedding_dim, collection_name="milvus_vector")
        # 初始化文档chunk关系数据库
        chunk_store = MilvusDocstore(client=client, collection_name="milvus_chunk")

        # Step2在线问题答复,初始化检索器
        text_retriever = Retriever(
            vector_store=vector_store,
            document_store=chunk_store,
            embed_func=emb.embed_documents,
            k=1,
            score_threshold=score_threshold,
        )
        # 配置text生成text大模型chain，具体ip端口请根据实际情况适配修改
        llm = Text2TextLLM(
            base_url=llm_url,
            model_name=model_name,
            client_param=ClientParam(use_http=True, timeout=60),
        )

        def process_query(input_string: str) -> str:
            text2text_chain = SingleText2TextChain(retriever=text_retriever, llm=llm)
            # 知识问答
            res = text2text_chain.query(input_string)
            # 打印结果
            logger.info(res)
            return f"{res}"

        results = []
        batch_size = len(query) // num_threads
        if len(query) % num_threads != 0:
            batch_size += 1
        batchs = [query[i : i + batch_size] for i in range(0, len(query), batch_size)]

        threads = []
        for batch in batchs:

            def process_batch(batch):
                batch_results = []
                for s in batch:
                    batch_results.append(process_query(s))
                return batch_results

            thread = ThreadWithResult(target=process_batch, args=(batch,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
            results.extend(thread.result)

        return results

    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(stack_trace)
        raise e


if __name__ == "__main__":
    rag_demo_query()
