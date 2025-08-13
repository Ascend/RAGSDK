# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
"""
向量数据库的工厂类，用于生产mxrag的向量数据库
"""
from abc import ABC
from typing import Optional

from mx_rag.storage.vectorstore import VectorStore, OpenGaussDB
from mx_rag.storage.vectorstore import MindFAISS
from mx_rag.storage.vectorstore import MilvusDB


class VectorStorageError(Exception):
    """
    向量数据库错误
    """
    pass


class VectorStorageFactory(ABC):
    """
    功能描述:
        向量数据库的工厂方法类，用于生产mxrag的向量数据库

    Attributes:
        NPU_SUPPORT_VEC_TYPE 字典，用于映射向量数据库和对应的构造函数
    """
    NPU_SUPPORT_VEC_TYPE = {
        "opengauss_db": OpenGaussDB.create,
        "npu_faiss_db": MindFAISS.create,
        "milvus_db": MilvusDB.create
    }

    @classmethod
    def create_storage(cls, **kwargs) -> Optional[VectorStore]:
        """
        功能描述:
            构造vector storage

        Args:
            kwargs: Dict[str, Any] 构造向量数据库的参数
        Return:
            vector_store: VectorStore 返回的构造向量数据库实例
        Raises:
            KeyError: 键值不存在
            ValueError: 数据类型不匹配
        """
        if "vector_type" not in kwargs:
            raise VectorStorageError("The 'vector_type' parameter is required.")

        vector_type = kwargs.pop("vector_type")

        if not isinstance(vector_type, str):
            raise VectorStorageError("The 'vector_type' parameter must be of type str.")

        if vector_type not in cls.NPU_SUPPORT_VEC_TYPE:
            raise VectorStorageError(f"The specified 'vector_type' '{vector_type}' is not supported.")

        creator = cls.NPU_SUPPORT_VEC_TYPE.get(vector_type)
        try:
            vector_store = creator(**kwargs)
        except KeyError as e:
            raise VectorStorageError("A KeyError occurred while creating the vector store.") from e
        except Exception as e:
            raise VectorStorageError("An unexpected error occurred while constructing the vector store.") from e

        return vector_store
