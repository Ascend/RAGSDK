# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
"""
向量数据库的工厂类，用于生产mxrag的向量数据库
"""
from abc import ABC
from typing import Optional, Dict, Any

from mx_rag.storage.vectorstore import VectorStore
from mx_rag.storage.vectorstore import MindFAISS
from mx_rag.storage.vectorstore import MilvusDB


class VectorStorageFactory(ABC):
    """
    功能描述:
        向量数据库的工厂方法类，用于生产mxrag的向量数据库

    Attributes:
        NPU_SUPPORT_VEC_TYPE 字典，用于映射向量数据库和对应的构造函数
    """
    NPU_SUPPORT_VEC_TYPE = {
        "npu_faiss_db": MindFAISS.create,
        "milvus_db": MilvusDB.create
    }

    @classmethod
    def create_storage(cls, vector_config: Dict[str, Any]) -> Optional[VectorStore]:
        """
        功能描述:
            构造vector storage

        Args:
            vector_config: Dict[str, Any] 构造向量数据库的参数
        Return:
            vector_store: VectorStore 返回的构造向量数据库实例
        Raises:
            KeyError: 键值不存在
            ValueError: 数据类型不匹配
        """
        if "vector_type" not in vector_config:
            raise KeyError("need vector_type param. ")

        vector_type = vector_config.pop("vector_type")

        if not isinstance(vector_type, str):
            raise ValueError("vector_type should be str type. ")

        if vector_type not in cls.NPU_SUPPORT_VEC_TYPE:
            raise KeyError(f"vector type is not support. {vector_type}")

        creator = cls.NPU_SUPPORT_VEC_TYPE.get(vector_type)
        vector_store = creator(**vector_config)
        return vector_store
