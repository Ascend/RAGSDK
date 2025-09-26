#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from typing import Callable
from sqlalchemy import Column, Integer, String, JSON, DateTime, text, Index, TEXT
from sqlalchemy.orm import declarative_base
from loguru import logger

Base = declarative_base()


class ChunkModel(Base):
    """
    Chunk 数据模型类，提供数据加密解密能力

    Attributes:
        chunk_id: 自增主键
        document_id: 关联文档ID
        document_name: 原始文档名称
        chunk_content: 文本块内容（支持加密）
        chunk_metadata: 元数据存储
        create_time: 记录创建时间（数据库自动生成）
    """
    __tablename__ = "chunks_table"

    chunk_id = Column(
        Integer,
        primary_key=True,
        comment="主键ID",
        autoincrement="auto"
    )
    document_id = Column(
        Integer,
        comment="文档ID"
    )
    document_name = Column(
        String(255),
        comment="文档名称"
    )
    chunk_content = Column(
        TEXT,
        comment="文本内容"
    )
    chunk_metadata = Column(
        JSON,
        comment="元数据"
    )
    create_time = Column(
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        comment="创建时间"
    )

    __table_args__ = (
        Index('ix_document_id', 'document_id'),
        Index('ix_create_time', 'create_time')
    )

    def __repr__(self) -> str:
        """调试用对象表示"""
        return (
            f"<Chunk(id={self.chunk_id}, "
            f"doc={self.document_id}, "
            f"content_length={len(self.chunk_content)})>"
        )