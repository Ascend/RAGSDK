# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import base64
import io
import os
from typing import List

import torch
from PIL import Image
from langchain_core.embeddings import Embeddings
from loguru import logger
from transformers import is_torch_npu_available
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from mx_rag.utils.common import validate_params, MAX_DEVICE_ID, EMBEDDING_TEXT_COUNT, \
    IMG_EMBEDDING_TEXT_LEN, validate_list_str, MB, GB, EMBEDDING_IMG_COUNT
from mx_rag.utils.file_check import SecFileCheck, SecDirCheck, safetensors_check

try:
    import torch_npu
    torch.npu.set_compile_mode(jit_compile=False)
except ImportError as e:
    logger.warning(f"Failed to import torch_npu: {e}. ImageEmbedding will run on cpu.")
except Exception as e:
    logger.error(f"Unexpected error while importing torch_npu: {e}. ImageEmbedding will run on cpu.")


_CLIP_MODELS = {
    "ViT-B-16": {
        "checkpoint": "clip_cn_vit-b-16.pt",
        "image_size": 224
    },
    "ViT-L-14": {
        "checkpoint": "clip_cn_vit-l-14.pt",
        "image_size": 224
    },
    "ViT-L-14-336": {
        "checkpoint": "clip_cn_vit-l-14-336.pt",
        "image_size": 336
    },
    "ViT-H-14": {
        "checkpoint": "clip_cn_vit-h-14.pt",
        "image_size": 224},
    "RN50": {
        "checkpoint": "clip_cn_rn50.pt",
        "image_size": 224
    },
}


class ImageEmbedding(Embeddings):
    @validate_params(
        model_name=dict(validator=lambda x: x in _CLIP_MODELS,
                        message=f"not supported model: {_CLIP_MODELS.keys()}"),
        dev_id=dict(validator=lambda x: 0 <= x <= MAX_DEVICE_ID, message="param value range [0, 63]")
    )
    def __init__(self, model_name: str, model_path: str, dev_id: int = 0):
        self.model_name = model_name
        self.model_path = model_path
        SecDirCheck(self.model_path, 10 * GB).check()
        safetensors_check(model_path)
        # 检查模型文件是否已就绪
        SecFileCheck(os.path.join(self.model_path, _CLIP_MODELS[self.model_name]['checkpoint']), 10 * GB).check()

        self.device = "cpu"
        try:
            if is_torch_npu_available():
                self.device = f'npu:{dev_id}'
        except ImportError:
            logger.warning('unable to import torch_npu, please check if torch_npu is properly installed. '
                           'currently running on cpu.')
        import cn_clip.clip as cnclip
        from cn_clip.clip import load_from_name
        self.model, self.preprocess = load_from_name(self.model_name, self.device, self.model_path)
        self.tokenizer = cnclip
        self.model.eval()

    @staticmethod
    def create(**kwargs):
        if "model_path" not in kwargs or not isinstance(kwargs.get("model_path"), str):
            logger.error("model_path param error. ")
            return None

        return ImageEmbedding(**kwargs)

    @validate_params(
        texts=dict(validator=lambda x: validate_list_str(x, [1, EMBEDDING_TEXT_COUNT], [1, IMG_EMBEDDING_TEXT_LEN]),
                   message="param must meets: Type is List[str], "
                           "list length range [1, 1000 * 1000], str length range [1, 256]"))
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        text = self.tokenizer.tokenize(texts, context_length=52).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text).detach()
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().tolist()

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= IMG_EMBEDDING_TEXT_LEN,
                  message=f"param must be str, and length range [1, {IMG_EMBEDDING_TEXT_LEN}]"))
    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_documents([text])
        if not embeddings:
            raise ValueError("embedding text failed")

        return embeddings[0]

    @validate_params(
        images=dict(
            validator=lambda x: (isinstance(x, list) and validate_list_str(x, [1, EMBEDDING_IMG_COUNT], [1, 10 * MB])),
            message=f"param must meets: Type is List[str], list length range [1, {EMBEDDING_IMG_COUNT}],"
                    f" str length range [1, {10 * MB}]"))
    def embed_images(self, images: List[str]) -> List[List[float]]:
        image_features = []
        batch_size = 32

        for start_idx in range(0, len(images), batch_size):
            batch_images = images[start_idx: start_idx + batch_size]
            tensors_batch = []
            for image in batch_images:
                blob = base64.b64decode(image)
                with Image.open(io.BytesIO(blob)) as img:
                    tensors_batch.append(self.preprocess(img).detach())
            tensors_batch = torch.stack(tensors_batch).to(self.device)
            with torch.no_grad():
                batch_image_features = self.model.encode_image(tensors_batch).detach()
                # 归一化
                batch_image_features = batch_image_features / batch_image_features.norm(p=2, dim=-1, keepdim=True)
                image_features.extend(batch_image_features.cpu().tolist())

        if not image_features:
            raise Exception("embedding image failed")

        return image_features
