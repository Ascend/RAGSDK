# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
"""
reranker的工厂类，用于生产mxrag的reranker
"""
from abc import ABC
from typing import Dict, Any, Callable
from loguru import logger

from mx_rag.reranker.reranker import Reranker
from mx_rag.reranker.local import LocalReranker
from mx_rag.reranker.service import TEIReranker


class RerankerFactory(ABC):
    """
    功能描述:
        reranker的工厂方法类，用于生产mxrag的reranker

    Attributes:
        NPU_SUPPORT_RERANKER 字典，用于映射reranker和对应的构造函数
    """
    NPU_SUPPORT_RERANKER: Dict[str, Callable[[Dict[str, Any]], Reranker]] = {
        "local_reranker": LocalReranker.create,
        "tei_reranker": TEIReranker.create
    }

    @classmethod
    def create_reranker(cls, **kwargs):
        """
        功能描述:
            构造vector storage

        Args:
            kwargs: Dict[str, Any] 构造reranker的参数
        Return:
            similarity: Reranker 返回的reranker的实例
        Raises:
            KeyError: 键值不存在
            ValueError: 数据类型不匹配
        """
        if "similarity_type" not in kwargs:
            logger.error("need similarity_config param. ")
            return None

        similarity_type = kwargs.pop("similarity_type")

        if not isinstance(similarity_type, str):
            logger.error("similarity_type should be str type. ")
            return None

        if similarity_type not in cls.NPU_SUPPORT_RERANKER:
            logger.error(f"similarity_type is not support. {similarity_type}")
            return None

        creator = cls.NPU_SUPPORT_RERANKER.get(similarity_type)

        try:
            similarity = creator(**kwargs)
        except KeyError:
            logger.error(f"create reranker key error")
            return None
        except Exception:
            logger.error(f"exception occurred while constructing reranker")
            return None

        return similarity
