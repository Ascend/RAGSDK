# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from pathlib import Path
from typing import List

import PIL
import numpy as np
import torch

from PIL import Image
from loguru import logger
from transformers import AutoProcessor, AutoModel, is_torch_npu_available

from mx_rag.embedding.embedding import Embedding
from mx_rag.utils.file_check import FileCheck, SecFileCheck

try:
    import torch_npu
    torch.npu.set_compile_mode(jit_compile=False)
except Exception as e:
    logger.warning(f"import torch_npu failed:{e}, img_embedding will running on cpu")


class ImageEmbedding(Embedding):
    SUPPORT_IMG_TYPE = (".jpg", ".png")
    MAX_IMAGE_SIZE = 100 * 1024 * 1024
    TEXT_COUNT = 1000 * 1000
    IMAGE_COUNT = 1000
    TEXT_LEN = 256

    def __init__(self, model_path: str, dev_id: int = 0, use_fp16: bool = True):
        self.model_path = model_path
        FileCheck.dir_check(self.model_path)

        self.use_fp16 = use_fp16
        self.model = AutoModel.from_pretrained(self.model_path)
        self.preprocess = AutoProcessor.from_pretrained(self.model_path)

        if self.use_fp16:
            self.model = self.model.half()

        try:
            if is_torch_npu_available():
                self.model.to(f'npu:{dev_id}')
        except ImportError:
            logger.warning('unable to import torch_npu, please check if torch_npu is properly installed. '
                           'currently running on cpu.')

    @staticmethod
    def create(**kwargs):
        if "model_path" not in kwargs or not isinstance(kwargs.get("model_path"), str):
            raise KeyError("model_path param error. ")

        return ImageEmbedding(**kwargs)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.array([])
        elif len(texts) > self.TEXT_COUNT:
            logger.error(f'texts list length must less than {self.TEXT_COUNT}')
            return np.array([])

        for text in texts:
            if len(text) > self.TEXT_LEN:
                logger.error(f"text len can not greater than {self.TEXT_LEN}")
                return np.array([])

        inputs = self.preprocess(text=texts, padding=True, return_tensors="pt", max_length=512).to(self.model.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs).detach().cpu()
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return text_features.numpy()

    def embed_images(self, images: List[str]) -> np.ndarray:
        image_features = []

        if len(images) == 0:
            return np.array([])
        elif len(images) > self.IMAGE_COUNT:
            logger.error(f'texts list length must less than {self.IMAGE_COUNT}')
            return np.array([])

        for image in images:
            FileCheck.check_path_is_exist_and_valid(image)
            SecFileCheck(image, self.MAX_IMAGE_SIZE).check()
            if Path(image).suffix not in self.SUPPORT_IMG_TYPE:
                raise TypeError(f"embed img:[{image}] failed because the file type not be supported")

            fi = PIL.Image.open(image)
            inputs = self.preprocess(images=fi, return_tensors="pt").to(self.model.device)
            fi.close()
            image_feature = self.model.get_image_features(**inputs)
            image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)

            image_features.extend(image_feature.detach().cpu().numpy())

        return np.stack(image_features)
