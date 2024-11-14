# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import base64
import io
import os
import json
from typing import List

import torch
from PIL import Image
from langchain_core.embeddings import Embeddings
from loguru import logger
from transformers import AutoProcessor, AutoModel, is_torch_npu_available
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from mx_rag.utils.common import validate_params, MAX_DEVICE_ID, EMBBEDDING_TEXT_COUNT, \
    IMG_EMBBEDDING_TEXT_LEN, validata_list_str, MB, EMBBEDDING_IMG_COUNT, BOOL_TYPE_CHECK_TIP
from mx_rag.utils.file_check import FileCheck, SecFileCheck

try:
    import torch_npu
    torch.npu.set_compile_mode(jit_compile=False)
except ImportError as e:
    logger.warning(f"Failed to import torch_npu: {e}. ImageEmbedding will run on cpu.")
except Exception as e:
    logger.error(f"Unexpected error while importing torch_npu: {e}. ImageEmbedding will run on cpu.")


def _preprocess_image(blob, image_size):
    blob = base64.b64decode(blob)
    img_transform = Compose(
        [
            Resize(image_size, interpolation=BICUBIC),
            CenterCrop(image_size),
            lambda x: x.convert('RGB'),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    with Image.open(io.BytesIO(blob)) as img:
        return img_transform(img)


class ImageEmbedding(Embeddings):
    @validate_params(
        dev_id=dict(validator=lambda x: 0 <= x <= MAX_DEVICE_ID, message="param value range [0, 63]"),
        use_fp16=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
    )
    def __init__(self, model_path: str, dev_id: int = 0, use_fp16: bool = True):
        self.model_path = model_path
        FileCheck.dir_check(self.model_path)

        self.use_fp16 = use_fp16
        self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.device = "cpu"

        config_file = os.path.join(self.model_path, 'config.json')
        SecFileCheck(config_file, 1 * MB).check()
        with open(config_file, 'r') as f:
            config = json.load(f)
            if 'vision_config' not in config or 'image_size' not in config['vision_config']:
                raise KeyError('wrong config.json: vision_config not found or image_size not found')
            self.image_size = config['vision_config']['image_size']

        if self.use_fp16:
            self.model = self.model.half()

        try:
            if is_torch_npu_available():
                self.device = f'npu:{dev_id}'
                self.model.to(self.device)
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
            validator=lambda x: (isinstance(x, list) and validata_list_str(x, [1, EMBBEDDING_IMG_COUNT], [1, 10*MB])),
            message=f"param must meets: Type is List[str], list length range [1, {EMBBEDDING_IMG_COUNT}],"
                    f" str length range [1, {10 * MB}]"))
    def embed_images(self, images: List[str]) -> List[List[float]]:
        image_features = []
        batch_size = 32

        for start_idx in range(0, len(images), batch_size):
            batch_images = images[start_idx: start_idx + batch_size]
            tensors_batch = []
            for image in batch_images:
                tensors_batch.append(_preprocess_image(image, self.image_size).detach())
            tensors_batch = torch.stack(tensors_batch).to(self.device)
            with torch.no_grad():
                batch_image_features = self.model.get_image_features(pixel_values=tensors_batch)
                # 归一化
                batch_image_features = batch_image_features / batch_image_features.norm(p=2, dim=-1, keepdim=True)
                image_features.extend(batch_image_features.cpu().numpy().tolist())

        if not image_features:
            raise Exception("embedding image failed")

        return image_features
