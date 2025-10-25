#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import io
import os
from abc import ABC
import zipfile
import psutil
import base64

from loguru import logger
from PIL import Image

from mx_rag.llm import Img2TextLLM
from mx_rag.utils import file_check
from mx_rag.utils.common import validate_params, STR_TYPE_CHECK_TIP_1024, MAX_IMAGE_PIXELS, MIN_IMAGE_WIDTH, \
    MIN_IMAGE_HEIGHT, MAX_BASE64_SIZE


class BaseLoader(ABC):
    MAX_SIZE = 100 * 1024 * 1024
    MAX_PAGE_NUM = 1000
    MAX_WORD_NUM = 500000
    MAX_FILE_CNT = 1024
    MAX_NESTED_DEPTH = 10

    @validate_params(
        file_path=dict(validator=lambda x: isinstance(x, str) and 0 < len(x) <= 1024, message=STR_TYPE_CHECK_TIP_1024),
    )
    def __init__(self, file_path):
        self.file_path = file_path
        self.multi_size = 5
        file_check.SecFileCheck(self.file_path, self.MAX_SIZE).check()

    def _is_zip_bomb(self):
        try:
            with zipfile.ZipFile(self.file_path, "r") as zip_ref:
                # 检查点1：检查文件个数，文件个数大于预期值时上报异常退出
                file_count = len(zip_ref.infolist())
                if file_count >= self.MAX_FILE_CNT * self.multi_size:
                    logger.error(f'zip file ({self.file_path}) contains {file_count} files, exceed '
                                 f'the limit of {self.MAX_FILE_CNT * self.multi_size}')
                    return True
                # 检查点2：检查第一层解压文件总大小，总大小超过设定的上限值
                total_uncompressed_size = sum(zinfo.file_size for zinfo in zip_ref.infolist())
                if total_uncompressed_size > self.MAX_SIZE * self.multi_size:
                    logger.error(f"zip file '{self.file_path}' uncompressed size is {total_uncompressed_size} bytes"
                                 f"exceeds the limit of {self.MAX_SIZE * self.multi_size} bytes, Potential ZIP bomb")
                    return True

                # 检查点3：检查第一层解压文件总大小，磁盘剩余空间-文件总大小<200M
                remain_size = psutil.disk_usage(os.getcwd()).free
                if remain_size - total_uncompressed_size < self.MAX_SIZE * 2:
                    logger.error(f'zip file ({self.file_path}) uncompressed size is {total_uncompressed_size} bytes'
                                 f' only {remain_size} bytes of disk space available')
                    return True

            # 嵌套深度检测
            with open(self.file_path, "rb") as f:
                data = f.read()
                depth = self._check_nested_depth(data)
                if depth > self.MAX_NESTED_DEPTH:
                    logger.error(f"{self.file_path} nested depth exceeds limit {self.MAX_NESTED_DEPTH}")
                    return True

            return False
        except zipfile.BadZipfile as e:
            logger.error(f"The provided path '{self.file_path}' is not a valid zip file or is corrupted: {e}")
            return True
        except Exception as e:
            logger.error(f"Unexpected error occurred while checking zip bomb: {e}")
            return True

    def _check_nested_depth(self, data: bytes, current_depth=1) -> int:
        """递归检查嵌套深度"""
        if current_depth > self.MAX_NESTED_DEPTH:
            return current_depth

        max_depth = current_depth
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
            for zinfo in zf.infolist():
                if zinfo.file_size == 0:
                    continue

                file_data = self._read_file_from_zip(zf, zinfo)
                if file_data is None:
                    continue

                depth = self._process_file(file_data, current_depth)
                max_depth = max(max_depth, depth)
                if max_depth > self.MAX_NESTED_DEPTH:
                    return max_depth

        except zipfile.BadZipfile as e:
            logger.error(f"The provided path '{self.file_path}' is not a valid zip file or is corrupted: {e}")

        return max_depth

    def _read_file_from_zip(self, zf: zipfile.ZipFile, zinfo: zipfile.ZipInfo):
        """从ZIP文件中读取文件数据"""
        try:
            return zf.read(zinfo)
        except zipfile.BadZipfile as e:
            logger.warning(f"Error processing {zinfo.filename}: {e}")
            return None

    def _process_file(self, file_data: bytes, current_depth: int) -> int:
        """处理文件数据"""
        if b'PK\x03\x04' in file_data:
            return self._check_nested_depth(file_data, current_depth + 1)
        else:
            return current_depth

    def _verify_image_size(self, image_bytes):
        """Verify if the image dimensions are within acceptable limits."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                width, height = img.size
                total_pixels = width * height
                if total_pixels > MAX_IMAGE_PIXELS:
                    logger.warning(f"Image too large: {width}x{height} pixels. Skipping.")
                    return False
                elif width < MIN_IMAGE_WIDTH and height < MIN_IMAGE_HEIGHT:
                    logger.warning(f"Image too small: {width}x{height} pixels. Skipping.")
                    return False

                return True
        except OSError as err:
            logger.error(f"Failed to open image file: {err}")
            return False
        except ValueError as err:
            logger.error(f"Invalid image data: {err}")
            return False
        except MemoryError as err:
            logger.error(f"Insufficient memory to process image: {err}")
            return False
        except Exception as err:
            logger.warning(f"Failed to verify image size: {err}")
            return False

    def _convert_to_base64(self, image_data,
                           max_base64_size=MAX_BASE64_SIZE,  # base64最大长度
                           max_iterations=10):
        """
        将图片转为Base64编码，并控制大小不超过 max_base64_size。
        - 如果原始数据直接符合要求，直接返回；
        - 否则按比例缩小分辨率，直到符合大小要求。
        """
        raw_b64_len = len(image_data) * 4 // 3
        if raw_b64_len <= max_base64_size:
            return base64.b64encode(image_data).decode("utf-8")

        with Image.open(io.BytesIO(image_data)) as image:
            image = image.convert("RGB")

            # 计算缩放比例
            ratio = (max_base64_size / raw_b64_len) ** 0.5
            new_w = int(image.width * ratio)
            new_h = int(image.height * ratio)

            resized_img = image.resize((new_w, new_h), Image.LANCZOS)

            buffer = io.BytesIO()
            resized_img.save(buffer, format="JPEG", quality=90)  # 默认高质量JPEG
            compressed_data = buffer.getvalue()

            return base64.b64encode(compressed_data).decode("utf-8")

    def _interpret_image(self, image_data, vlm: Img2TextLLM):
        img_base64 = self._convert_to_base64(image_data)
        if self._verify_image_size(image_data) is False:
            logger.warning("image size is invalid")
            img_base64, img_summary = "", ""
            return img_base64, img_summary
        # vllm解析图像
        image_url = {"url": f"data:image/jpeg;base64,{img_base64}"}
        image_summary = vlm.chat(image_url=image_url)
        if image_summary == "":
            img_base64 = ""
            logger.warning("image summary func exec failed")
        return img_base64, image_summary
