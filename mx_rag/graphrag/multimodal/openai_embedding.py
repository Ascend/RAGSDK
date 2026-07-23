#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

RAGSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import base64
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai._types import NOT_GIVEN, NotGiven


class OpenAIEmbedding:
    """统一的文本和图像 Embedding 接口，基于 vLLM Chat Embeddings API。"""

    SUPPORTED_IMAGE_FORMATS = frozenset({"jpg", "jpeg", "png", "gif", "bmp", "webp"})
    DEFAULT_INSTRUCTION = "Represent the user's input."

    def __init__(
        self,
        model_name: str,
        client: Optional[OpenAI] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_concurrent: int = 8,
    ):
        self.model_name = model_name
        self.max_concurrent = max_concurrent

        # 优先使用 client，其次 url+api_key 创建
        if client:
            self._client = client
            self.url = str(client.base_url)
        elif url and api_key:
            self._client = None
            self.url = url
            self._api_key = api_key
        else:
            raise ValueError("必须提供 client，或同时提供 url 和 api_key")

    # ── 客户端（懒加载） ──
    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self.url)
        return self._client

    # ── 静态工具方法 ──
    @staticmethod
    def create_chat_embeddings(
        client: OpenAI,
        *,
        messages: list[ChatCompletionMessageParam],
        model: str,
        encoding_format: Literal["base64", "float"] | NotGiven = NOT_GIVEN,
        continue_final_message: bool = False,
        add_special_tokens: bool = False,
    ) -> CreateEmbeddingResponse:
        """vLLM Chat Embeddings API（OpenAI Embeddings API 的扩展）。"""
        return client.post(
            "/embeddings",
            cast_to=CreateEmbeddingResponse,
            body={
                "messages": messages,
                "model": model,
                "encoding_format": encoding_format,
                "continue_final_message": continue_final_message,
                "add_special_tokens": add_special_tokens,
            },
        )

    @staticmethod
    def encode_image_to_base64(image_path: str, max_size_mb: float = 10.0) -> str:
        """将本地图片文件编码为 data:image 格式的 Base64 字符串。"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        file_size = os.path.getsize(image_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise ValueError(f"图片文件大小 {file_size / 1024 / 1024:.2f}MB 超过限制 {max_size_mb}MB")

        _, ext = os.path.splitext(image_path)
        image_format = ext.lower().lstrip(".") or "png"

        if image_format not in OpenAIEmbedding.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"不支持的图片格式: {image_format}")

        with open(image_path, "rb") as f:
            base64_encoded = base64.b64encode(f.read()).decode("utf-8")

        return f"data:image/{image_format};base64,{base64_encoded}"

    # ── 构建消息 ──
    @staticmethod
    def _build_image_messages(image_path: str) -> list[ChatCompletionMessageParam]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": OpenAIEmbedding.DEFAULT_INSTRUCTION}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": OpenAIEmbedding.encode_image_to_base64(image_path)},
                    },
                    {"type": "text", "text": ""},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
            },
        ]

    @staticmethod
    def _build_text_messages(text: str) -> list[ChatCompletionMessageParam]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": OpenAIEmbedding.DEFAULT_INSTRUCTION}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
            },
        ]

    # ── 并发执行 ──
    def _embed_concurrent(self, items: list, build_messages_fn) -> list:
        """通过线程池并发请求，保持输入顺序，N 个输入返回 N 个 embedding。"""
        if not items:
            return []

        client = self._get_client()
        results = [None] * len(items)

        def _embed(item):
            messages = build_messages_fn(item)
            response = self.create_chat_embeddings(
                client,
                messages=messages,
                model=self.model_name,
                encoding_format="float",
                continue_final_message=True,
                add_special_tokens=True,
            )
            return response.data[0].embedding

        max_workers = min(len(items), self.max_concurrent)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_embed, item): idx for idx, item in enumerate(items)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return [r for r in results if r is not None]

    # ── 公开接口 ──
    def run_image(self, image_paths: List[str]) -> list:
        """每张图片独立构建 conversation，返回各图片的 embedding。"""
        return self._embed_concurrent(image_paths, self._build_image_messages)

    def run_text(self, texts: List[str]) -> list:
        """每段文本独立构建 conversation，返回各文本的 embedding。"""
        return self._embed_concurrent(texts, self._build_text_messages)
