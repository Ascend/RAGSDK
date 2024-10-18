# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import base64
import io
from typing import List

import torch
from PIL import Image
from langchain_core.embeddings import Embeddings
from loguru import logger
from transformers import AutoProcessor, AutoModel, is_torch_npu_available

from mx_rag.utils.common import validate_params, MAX_DEVICE_ID, EMBBEDDING_TEXT_COUNT, \
    IMG_EMBBEDDING_TEXT_LEN, validata_list_str, MB, EMBBEDDING_IMG_COUNT, BOOL_TYPE_CHECK_TIP
from mx_rag.utils.file_check import FileCheck

try:
    import torch_npu

    torch.npu.set_compile_mode(jit_compile=False)
except ImportError as e:
    logger.warning(f"Failed to import torch_npu: {e}. ImageEmbedding will run on cpu.")
except Exception as e:
    logger.error(f"Unexpected error while importing torch_npu: {e}. ImageEmbedding will run on cpu.")


class ImageEmbedding(Embeddings):

    @validate_params(
        dev_id=dict(validator=lambda x: 0 <= x <= MAX_DEVICE_ID, message="param value range [0, 63]"),
        use_fp16=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
    )
    def __init__(self, model_path: str, dev_id: int = 0, use_fp16: bool = True):
        self.model_path = model_path
        FileCheck.dir_check(self.model_path)

        self.use_fp16 = use_fp16
        self.model = AutoModel.from_pretrained(self.model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path)

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
            logger.error("model_path param error. ")
            return None

        return ImageEmbedding(**kwargs)

    @validate_params(
        texts=dict(validator=lambda x: validata_list_str(x, [1, EMBBEDDING_TEXT_COUNT], [1, IMG_EMBBEDDING_TEXT_LEN]),
                   message="param must meets: Type is List[str], "
                           "list length range [1, 1000 * 1000], str length range [1, 256]"))
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.processor(text=texts, padding=True, return_tensors="pt", max_length=512).to(self.model.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs).detach().cpu()
            text_features_norm = text_features.norm(p=2, dim=-1, keepdim=True)
            contains_zero = torch.any(torch.eq(text_features_norm, 0))
            if contains_zero:
                raise ValueError("contains zero, can not be divide")
            text_features = text_features / text_features_norm

        return text_features.tolist()

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_documents([text])
        if not embeddings:
            raise ValueError("embedding text failed")

        return embeddings[0]

    @validate_params(
        images=dict(
            validator=lambda x: validata_list_str(x, [1, EMBBEDDING_IMG_COUNT], [1, 100*MB]),
            message="param must meets: Type is List[str], list length range [0, 1000], str length range [1, 100*MB]"))
    def embed_images(self, images: List[str]) -> List[List[float]]:
        image_features = []

        for image_content in images:
            with Image.open(io.BytesIO(base64.b64decode(image_content))) as fi:
                inputs = self.processor(images=fi, return_tensors="pt").to(self.model.device)

            image_feature = self.model.get_image_features(**inputs)
            image_feature_norm = image_feature.norm(p=2, dim=-1, keepdim=True)
            contains_zero = torch.any(torch.eq(image_feature_norm, 0))
            if contains_zero:
                raise ValueError("contains zero, can not be divide")
            image_feature = image_feature / image_feature_norm

            image_features.append(image_feature)

        if not image_features:
            raise Exception("embedding image failed")

        return torch.cat(image_features).tolist()
