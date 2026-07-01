# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

# pylint: disable=duplicate-code

import argparse
import traceback

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from loguru import logger
from pymilvus import MilvusClient

from mx_rag.document import LoaderMng
from mx_rag.document.loader import DocxLoader, PdfLoader
from mx_rag.embedding.service import TEIEmbedding
from mx_rag.knowledge import KnowledgeDB
from mx_rag.knowledge.handler import upload_files
from mx_rag.knowledge.knowledge import KnowledgeStore
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.storage.vectorstore import MilvusDB
from mx_rag.utils import ClientParam


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_default_metavar_for_optional(self, action):
        return action.type.__name__

    def _get_default_metavar_for_positional(self, action):
        return action.type.__name__


def rag_demo_upload():
    parse = argparse.ArgumentParser(formatter_class=CustomFormatter)
    parse.add_argument(
        "--embedding_url",
        type=str,
        help="向量化服务地址，例如: http://127.0.0.1:8080/v1/embeddings",
    )
    parse.add_argument("--white_path", type=str, nargs="+", default=["/home"], help="文件白名单路径")
    parse.add_argument(
        "--file_path",
        type=str,
        action="append",
        help="要上传的文件路径，需在白名单路径下, 支持多个文件同时传入，例如：--file_path /home/file1.txt  --file_path /home/file2.pdf",
    )

    args = parse.parse_args()

    try:
        # 离线构建知识库,首先注册文档处理器
        loader_mng = LoaderMng()
        # 加载文档加载器，可以使用mxrag自有的，也可以使用langchain的
        loader_mng.register_loader(loader_class=TextLoader, file_types=[".txt", ".md"])
        loader_mng.register_loader(loader_class=PdfLoader, file_types=[".pdf"])
        loader_mng.register_loader(loader_class=DocxLoader, file_types=[".docx"])
        # 加载文档切分器，使用langchain的
        loader_mng.register_splitter(
            splitter_class=RecursiveCharacterTextSplitter,
            file_types=[".pdf", ".docx", ".txt", ".md"],
            splitter_params={
                "chunk_size": 200,
                "chunk_overlap": 30,
                "keep_separator": False,
            },
        )
        # 初始化向量模型对象
        emb = TEIEmbedding(url=args.embedding_url, client_param=ClientParam(use_http=True))
        embedding_dim = len(emb.embed_query("你好"))
        client = MilvusClient("./vector.db")

        vector_store = MilvusDB.create(client=client, x_dim=embedding_dim)
        # 初始化文档chunk关系数据库
        chunk_store = SQLiteDocstore(db_path="./chunk.db")
        # 初始化知识管理关系数据库
        knowledge_store = KnowledgeStore(db_path="./knowledge.db")
        # 添加知识库
        knowledge_store.add_knowledge("test", "Default", "admin")
        # 初始化知识库管理
        knowledge_db = KnowledgeDB(
            knowledge_store=knowledge_store,
            chunk_store=chunk_store,
            vector_store=vector_store,
            knowledge_name="test",
            white_paths=args.white_path,
            user_id="Default",
        )
        # 上传知识文件
        upload_files(
            knowledge=knowledge_db,
            files=args.file_path,
            loader_mng=loader_mng,
            embed_func=emb.embed_documents,
            force=True,
        )

        # 检验文件是否上传成功
        documents = [document.document_name for document in knowledge_db.get_all_documents()]
        logger.info(documents)
    except Exception:
        stack_trace = traceback.format_exc()
        logger.error(stack_trace)


if __name__ == "__main__":
    rag_demo_upload()
