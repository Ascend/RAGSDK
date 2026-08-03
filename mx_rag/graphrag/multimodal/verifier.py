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

import math
from typing import List

import numpy as np
from loguru import logger
from tqdm import tqdm

from mx_rag.graphrag.multimodal.multimodal_config import MultimodalConfig
from mx_rag.graphrag.multimodal.openai_embedding import OpenAIEmbedding


class CaptionVerifier:
    def __init__(self, config: MultimodalConfig):
        self._config = config
        self._truncate_dim = config.truncate_dim
        self._filter_type = config.filter_type
        self._device = config.device
        self._model = None
        self._openai_embed = None
        if config.emb_server_url:
            self._openai_embed = OpenAIEmbedding(model_name=config.emb_model_name, url=config.emb_server_url)
            logger.info(
                "Using OpenAIEmbedding service at {} with model {}", config.emb_server_url, config.emb_model_name
            )
        elif config.emb_model_path:
            self._load_model(config.emb_model_path)

    def _load_model(self, model_path: str):
        try:
            from transformers import AutoModel
            import torch

            logger.info("Loading embedding model from {} on device {}", model_path, self._device)
            self._model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
            self._model.to(self._device)
            logger.info("Embedding model loaded successfully")
        except ImportError as e:
            raise RuntimeError(
                "transformers and torch are required for CaptionVerifier. "
                "Install them with: pip install transformers torch"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model from {model_path}: {e}") from e

    def verify(self, captions: List[dict]) -> List[dict]:
        if self._openai_embed is None and self._model is None:
            logger.warning("Embedding not configured, skipping verification")
            return captions
        if not captions:
            return captions
        logger.info("Verifying {} captions with filter_type={}", len(captions), self._filter_type)
        for idx, item in enumerate(tqdm(captions, desc="Verifying captions")):
            try:
                captions[idx]["text"] = self.process_text(item["id"], item["text"])
            except Exception as e:
                logger.warning("Failed to verify caption for {}: {}", item["id"], e)
        return captions

    def process_text(self, image_path: str, text: str) -> str:
        if not text or not text.strip():
            return ""
        chunks = self.chunk_text(text)
        chunks = [" ".join(c.split()) for c in chunks]
        chunks = [c for c in chunks if c]
        if not chunks:
            return ""
        similarities = self._compute_similarity(chunks, image_path)
        if similarities.size == 0:
            return "。".join(chunks)
        threshold_value = self._filter_option(similarities, self._filter_type)
        filtered_chunks = [chunk for chunk, sim in zip(chunks, similarities) if sim >= threshold_value]
        if not filtered_chunks:
            return "。".join(chunks)
        return "。".join(filtered_chunks)

    def _compute_similarity(self, chunks: List[str], image_path: str) -> np.ndarray:
        try:
            if self._openai_embed is not None:
                return self._compute_similarity_service(chunks, image_path)
            else:
                return self._compute_similarity_local(chunks, image_path)
        except Exception as e:
            logger.error("Failed to compute similarity: {}", e)
            return np.array([])

    def _compute_similarity_local(self, chunks: List[str], image_path: str) -> np.ndarray:
        text_embeddings = self._model.encode_text(
            sentences=chunks, truncate_dim=self._truncate_dim, show_progress_bar=False
        )
        image_embeddings = self._model.encode_image(
            [image_path], truncate_dim=self._truncate_dim, show_progress_bar=False
        )
        similarity = (image_embeddings @ text_embeddings.T).squeeze(0)
        return similarity

    def _compute_similarity_service(self, chunks: List[str], image_path: str) -> np.ndarray:
        text_embeddings = self._openai_embed.run_text(chunks)
        image_embeddings = self._openai_embed.run_image([image_path])
        text_emb = np.array(text_embeddings, dtype=np.float32)
        img_emb = np.array(image_embeddings, dtype=np.float32)
        if self._truncate_dim and self._truncate_dim < text_emb.shape[1]:
            text_emb = text_emb[:, : self._truncate_dim]
            img_emb = img_emb[:, : self._truncate_dim]
        similarity = (img_emb @ text_emb.T).squeeze(0)
        return similarity

    def _filter_option(self, similarities: np.ndarray, option: int) -> float:
        if option == 1:
            return self._filter_largest_percent(similarities)
        elif option == 2:
            return self._filter_by_std(similarities)
        elif option == 3:
            return self._filter_by_max_delta_segment(similarities)
        elif option == 4:
            return self._filter_by_max_delta_segment_with_std(similarities)
        return 0.0

    @staticmethod
    def _filter_largest_percent(similarities: np.ndarray, percent: int = 20) -> float:
        n_remove = int(len(similarities) * percent / 100)
        if n_remove == 0:
            return 0.0
        sorted_sims = sorted(similarities)
        return sorted_sims[n_remove]

    @staticmethod
    def _filter_by_std(similarities: np.ndarray, coef: float = 0.5) -> float:
        minimum = min(similarities)
        mean = sum(similarities) / len(similarities)
        sigma = math.sqrt(sum((s - mean) ** 2 for s in similarities) / len(similarities))
        return minimum + coef * sigma

    @staticmethod
    def _filter_by_max_delta_segment(similarities: np.ndarray, low: float = 0.1, high: float = 0.3) -> float:
        length = len(similarities)
        if length <= 2:
            return 0.0
        segment_start_index = max(1, int(low * length))
        segment_end_index = int(high * length)
        if segment_end_index < segment_start_index:
            return 0.0
        segment = sorted(similarities)[segment_start_index - 1 : segment_end_index + 1]
        if len(segment) <= 1:
            return 0.0
        max_delta = -1
        max_index = 0
        for i in range(len(segment) - 1):
            delta = abs(segment[i + 1] - segment[i])
            if delta > max_delta:
                max_delta = delta
                max_index = i
        threshold_index = max_index + 1
        return segment[threshold_index]

    @staticmethod
    def _filter_by_max_delta_segment_with_std(
        similarities: np.ndarray, low: float = 0.1, high: float = 0.3, coef: float = 0.1
    ) -> float:
        length = len(similarities)
        if length <= 2:
            return 0.0
        segment_start_index = max(1, int(low * length))
        segment_end_index = int(high * length)
        if segment_end_index < segment_start_index:
            return 0.0
        segment = sorted(similarities)[segment_start_index - 1 : segment_end_index + 1]
        if len(segment) <= 1:
            return 0.0
        max_delta = -1
        max_index = 0
        for i in range(len(segment) - 1):
            delta = abs(segment[i + 1] - segment[i])
            if delta > max_delta:
                max_delta = delta
                max_index = i
        threshold_value = segment[max_index]
        mean = sum(similarities) / len(similarities)
        sigma = math.sqrt(sum((s - mean) ** 2 for s in similarities) / len(similarities))
        threshold_value += coef * sigma
        return threshold_value

    @staticmethod
    def chunk_text(text: str) -> List[str]:
        if "。" in text:
            return text.split("。")
        else:
            return text.split(".")
