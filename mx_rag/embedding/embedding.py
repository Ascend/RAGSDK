# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from abc import ABC, abstractmethod
from typing import List


class Embedding(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_texts(self, texts: List[str]):
        """Embed docs."""

    def embed_images(self, images: List[str]):
        """Embed images."""
