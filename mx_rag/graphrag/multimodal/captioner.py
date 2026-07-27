#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

import json
import os
import re
import stat
from typing import Dict, List

from loguru import logger
from tqdm import tqdm

from paddle.base import libpaddle  # noqa: F401
from mx_rag.graphrag.multimodal.multimodal_config import (
    MultimodalConfig,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from mx_rag.graphrag.multimodal.vlm_inference import VLMInferenceEngine
from mx_rag.graphrag.prompts.multimodal_prompt import (
    MULTIMODAL_FINE_PROMPT_CN,
    MULTIMODAL_INIT_PROMPT_CN,
)


class ImageCaptioner:
    def __init__(self, config: MultimodalConfig):
        self._config = config
        self._suffix = SUPPORTED_IMAGE_EXTENSIONS
        self._loops = len(config.vlm_servers)
        self._num_workers_per_server = min(config.num_workers_per_server, 64)
        self._batch_size = min(config.batch_size, 64)
        self._generator = VLMInferenceEngine(config)
        self._prompts_dict: Dict[str, str] = {}
        self._vlm_result_dir: str = ""
        self._load_prompts()

    def _load_prompts(self):
        if (
            self._config.prompt_path
            and os.path.isfile(self._config.prompt_path)
            and self._config.prompt_path.endswith(".json")
        ):
            with open(self._config.prompt_path, "r", encoding="utf-8") as f:
                self._prompts_dict = json.loads(f.read())
            logger.info("Loaded multimodal prompts from {}", self._config.prompt_path)
        else:
            self._prompts_dict = {
                "init_prompt": MULTIMODAL_INIT_PROMPT_CN,
                "fine_prompt": MULTIMODAL_FINE_PROMPT_CN,
            }
            logger.info("Using built-in multimodal prompts (Chinese)")

    def set_vlm_result_dir(self, vlm_result_dir: str):
        self._vlm_result_dir = vlm_result_dir
        os.makedirs(self._vlm_result_dir, exist_ok=True)

    async def caption(self, image_list: List[str]) -> List[dict]:
        if not image_list:
            logger.warning("No images to caption")
            return []
        samples = self._filter_image_files(image_list)
        if not samples:
            logger.warning("No valid image files found after filtering")
            return []
        init_inputs = [[self._prompts_dict["init_prompt"], s] for s in samples]
        logger.info("Image captioning: {} images, {} loops", len(samples), self._loops)
        result_path = os.path.join(self._vlm_result_dir, f"{self._config.vlm_result_name}.json")
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        modes = stat.S_IWUSR | stat.S_IRUSR
        total_batches = (len(init_inputs) + self._batch_size - 1) // self._batch_size
        with os.fdopen(os.open(result_path, flags, modes), "w", encoding="utf-8") as f:
            for batch in tqdm(
                self._batch_generator(init_inputs, self._batch_size),
                total=total_batches,
                desc="Captioning batches",
            ):
                generate_texts = await self._generator.run(batch, idx=0)
                for i in range(1, self._loops):
                    fine_inputs = self._build_fine_inputs(batch, generate_texts)
                    if fine_inputs:
                        fine_results = await self._generator.run(fine_inputs, idx=i)
                        generate_texts += fine_results
                self._write_batch_results(f, batch, generate_texts)
        return self._load_captions(result_path)

    def _filter_image_files(self, image_list: List[str]) -> List[str]:
        return [str(s) for s in image_list if str(s).rsplit(".", maxsplit=1)[-1] in self._suffix]

    def _build_fine_inputs(self, batch: list, generate_texts: list) -> list:
        """构建精炼阶段的输入列表。"""
        fine_inputs = []
        for idx, s in enumerate(batch):
            if s[1].split(".")[-1] in self._suffix:
                prev_text = ""
                if idx < len(generate_texts) and generate_texts[idx] is not None:
                    prev_text = (
                        generate_texts[idx] if isinstance(generate_texts[idx], str) else str(generate_texts[idx])
                    )
                fine_inputs.append([prev_text + self._prompts_dict["fine_prompt"], s[1]])
        return fine_inputs

    def _write_batch_results(self, f, batch: list, generate_texts: list):
        """将批次结果写入文件。"""
        for idx, item in enumerate(generate_texts):
            if idx >= len(batch):
                break
            text = item if isinstance(item, str) else (str(item) if item is not None else "")
            text = self._replace_image_markdown(text)
            data = {
                "id": batch[idx][1],
                "text": text,
                "metadata": {"lang": "zh-CN"},
                "pattern": "vlm",
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.flush()

    @staticmethod
    def _load_captions(file_path: str) -> List[dict]:
        captions = []
        if not os.path.isfile(file_path):
            return captions
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("text"):
                        captions.append(data)
                except json.JSONDecodeError:
                    continue
        return captions

    @staticmethod
    def _batch_generator(lst, batch_size: int):
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]

    @staticmethod
    def _replace_image_markdown(text: str) -> str:
        pattern = r"!\[[^\]]*\]\([^\)]*\)"
        return re.sub(pattern, "", text)
