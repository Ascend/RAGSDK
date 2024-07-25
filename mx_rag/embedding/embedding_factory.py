# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
"""
embedding的工厂类，用于生产mxrag的embedding
"""
from abc import ABC
from typing import Dict, Any, Callable

from mx_rag.embedding import Embedding
from mx_rag.embedding.local import TextEmbedding, ImageEmbedding
from mx_rag.embedding.service import TEIEmbedding


class EmbeddingFactory(ABC):
    """
    功能描述:
        embedding的工厂方法类，用于生产mxrag的embedding

    Attributes:
        NPU_SUPPORT_EMB 字典，用于映射embedding和对应的构造函数
    """
    NPU_SUPPORT_EMB: Dict[str, Callable[[Dict[str, Any]], Embedding]] = {
        "local_text_embedding": TextEmbedding.create,
        "local_imags_embedding": ImageEmbedding.create,
        "tei_embedding": TEIEmbedding.create
    }

    @classmethod
    def create_embedding(cls, embedding_config: Dict[str, Any]) -> Embedding:
        """
        功能描述:
            构造embedding

        Args:
            embedding_config: Dict[str, Any] 构造embedding的参数
        Return:
            embedding: Embedding 返回的embedding的实例
        Raises:
            KeyError: 键值不存在
            ValueError: 数据类型不匹配
        """
        if "embedding_type" not in embedding_config:
            raise KeyError("need embedding_type param. ")

        embedding_type = embedding_config.pop("embedding_type")

        if not isinstance(embedding_type, str):
            raise ValueError("embedding_type should be str type. ")

        if embedding_type not in cls.NPU_SUPPORT_EMB:
            raise KeyError(f"embedding_type is not support. {embedding_type}")

        creator = cls.NPU_SUPPORT_EMB.get(embedding_type)
        embedding = creator(**embedding_config)
        return embedding
