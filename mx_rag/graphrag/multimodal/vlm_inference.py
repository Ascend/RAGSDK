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

import asyncio
from typing import List

from loguru import logger

from paddle.base import libpaddle  # noqa: F401
from mx_rag.graphrag.multimodal.multimodal_config import MultimodalConfig
from mx_rag.graphrag.multimodal.openai_embedding import OpenAIEmbedding
from mx_rag.llm import Img2TextLLM
from mx_rag.utils import ClientParam


class VLMInferenceEngine:
    """VLM 推理引擎：服务化模式（Img2TextLLM），统一返回 ``List[str]``。"""

    def __init__(self, config: MultimodalConfig):
        self._config = config
        self._K = min(config.num_workers_per_server, 64)
        self._timeout = config.timeout
        # _SERVERS[round] = 该轮可用服务地址（按 _K 复制做轮询负载均衡）
        self._SERVERS: List[List[str]] = []
        # _llm_pool[round] = 与 _SERVERS[round] 一一对应的 Img2TextLLM 客户端
        self._llm_pool: List[List[Img2TextLLM]] = []
        if not config.vlm_servers:
            raise ValueError("vlm_servers must be configured for service mode")
        self._init_service(config)

    # ── 服务模式初始化 ──
    def _init_service(self, config: MultimodalConfig):
        for idx, round_servers in enumerate(config.vlm_servers):
            replicated = list(round_servers) * self._K
            self._SERVERS.append(replicated)
            self._llm_pool.append(
                [
                    Img2TextLLM(
                        base_url=srv,
                        model_name=config.vlm_model_name,
                        client_param=ClientParam(use_http=True, timeout=self._timeout),
                    )
                    for srv in replicated
                ]
            )
        logger.info("VLM SERVERS: {}", str(self._SERVERS))

    # ── 统一入口：返回 List[str] ──
    async def run(self, batch: list, idx: int = 0) -> List[str]:
        return await self._infer_service(batch, idx)

    # ── 服务分支：并发调用 Img2TextLLM.chat ──
    async def _infer_service(self, batch: list, idx: int) -> List[str]:
        if idx >= len(self._SERVERS) or not self._SERVERS[idx]:
            logger.error("No available VLM servers for round {}", idx)
            return [""] * len(batch)
        llm_list = self._llm_pool[idx]
        tasks = [
            asyncio.to_thread(self._call_service, llm_list[i % len(llm_list)], sample[0], sample[1])
            for i, sample in enumerate(batch)
        ]
        responses = await asyncio.gather(*tasks)
        return [r if isinstance(r, str) else "" for r in responses]

    @staticmethod
    def _call_service(llm: Img2TextLLM, prompt: str, image_path: str) -> str:
        try:
            image_url = {"url": OpenAIEmbedding.encode_image_to_base64(image_path)}
            return llm.chat(image_url=image_url, prompt=prompt)
        except Exception as e:
            logger.error("VLM service call failed for {}: {}", image_path, e)
            return ""
