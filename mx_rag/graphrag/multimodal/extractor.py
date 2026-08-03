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

import asyncio
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import List, Optional

from loguru import logger

from mx_rag.graphrag.multimodal.captioner import ImageCaptioner
from mx_rag.graphrag.multimodal.multimodal_config import MultimodalConfig
from mx_rag.graphrag.multimodal.parser import DocumentParser
from mx_rag.graphrag.multimodal.verifier import CaptionVerifier

MAX_FILE_NUM = 1000


class MultimodalPrepare:
    def __init__(self, config: MultimodalConfig):
        if not isinstance(config, MultimodalConfig):
            raise ValueError("config must be an instance of MultimodalConfig")
        self._config = config
        self._output_folder = config.output_folder
        os.makedirs(self._output_folder, exist_ok=True)
        self._parser = DocumentParser(config)
        self._captioner = ImageCaptioner(config)
        vlm_result_dir = os.path.join(self._output_folder, "vlm_result")
        self._captioner.set_vlm_result_dir(vlm_result_dir)
        self._verifier: Optional[CaptionVerifier] = None
        if config.emb_model_path or config.emb_server_url:
            self._verifier = CaptionVerifier(config)

    def extract(self, file_list: List[str]) -> None:
        if not isinstance(file_list, list) or len(file_list) == 0:
            raise ValueError("file_list must be a non-empty list of str")
        if len(file_list) > MAX_FILE_NUM:
            raise ValueError(f"file_list length must be in range [1, {MAX_FILE_NUM}]")
        for f in file_list:
            if not os.path.isfile(f):
                raise FileNotFoundError(f"File not found: {f}")
        logger.info("Starting multimodal extraction for {} files", len(file_list))
        t_start = time.time()
        md_files, images = self._parser.parse(file_list)
        if not md_files:
            logger.warning("No markdown files produced by parser")
            return
        logger.info("Parser produced {} markdown files, {} image files", len(md_files), len(images))
        vlm_result_path = os.path.join(self._output_folder, "vlm_result", f"{self._config.vlm_result_name}.json")
        filter_extracted_images = self._filter_extracted_images(images, vlm_result_path)
        logger.info("Filtered extracted images: {} (remaining to process)", len(filter_extracted_images))
        if images:
            if filter_extracted_images:
                logger.info("Starting image captioning...")
                t2 = time.time()
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                self._captioner.caption(filter_extracted_images),
                            )
                            captions = future.result()
                    else:
                        captions = loop.run_until_complete(self._captioner.caption(filter_extracted_images))
                except RuntimeError:
                    captions = asyncio.run(self._captioner.caption(filter_extracted_images))
                t3 = time.time()
                logger.info("Image captioning completed in {:.2f}s", t3 - t2)
            else:
                logger.info("All images already processed, loading existing captions")
            captions = self._load_captions(vlm_result_path)
            logger.info("Loaded {} captions", len(captions))
            if self._verifier is not None:
                logger.info("Starting caption verification...")
                t4 = time.time()
                captions = self._verifier.verify(captions)
                t5 = time.time()
                logger.info("Caption verification completed in {:.2f}s", t5 - t4)
            else:
                logger.info("Caption verification skipped (emb_model_path not set)")
            if len(captions) != len(images):
                logger.warning(
                    "Caption count ({}) != image count ({}), merging with available captions",
                    len(captions),
                    len(images),
                )
            self._merge_caption(self._output_folder, captions, self._config.merge_type)
        else:
            logger.warning("No images found in {}, copying markdown files directly", self._output_folder)
            md_files_direct = list(Path(self._output_folder).rglob("*.md"))
            for mf in md_files_direct:
                dest = os.path.join(self._output_folder, mf.name)
                shutil.copyfile(str(mf), dest)
        t_end = time.time()
        logger.info("Multimodal extraction completed in {:.2f}s", t_end - t_start)

    def _filter_extracted_images(self, images_path: List[str], vlm_result_path: str) -> List[str]:
        if not os.path.isfile(vlm_result_path):
            return images_path
        extracted_ids = set()
        try:
            with open(vlm_result_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("text"):
                            extracted_ids.add(data["id"])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning("Failed to read VLM result file {}: {}", vlm_result_path, e)
            return images_path
        return [item for item in images_path if str(item) not in extracted_ids]

    def _merge_caption(self, path: str, captions: List[dict], merge_type: str = "replace"):
        md_files = list(Path(path).rglob("*.md"))
        for md_path in md_files:
            try:
                with open(md_path, "r", encoding="utf-8") as r_file:
                    content = r_file.read()
            except Exception as e:
                logger.warning("Failed to read markdown file {}: {}", md_path, e)
                continue
            if merge_type == "replace":
                for c in captions:
                    image_path = Path(c["id"])
                    if len(image_path.parts) >= 4 and image_path.parts[-4] == md_path.parts[-3]:
                        filename = image_path.parts[-1]
                        pattern = r"!\[[^\]]*\]\(images/" + re.escape(filename) + r"\)"
                        new_text = c["text"]
                        content = re.sub(pattern, new_text, content)
            elif merge_type == "append":
                for c in captions:
                    image_path = Path(c["id"])
                    if len(image_path.parts) >= 4 and image_path.parts[-4] == md_path.parts[-3]:
                        content += "\n"
                        content += "# " + c["text"].replace("\n\n", "\n") + "\n\n"
            dest_path = os.path.join(self._output_folder, md_path.name)
            try:
                with open(dest_path, "w", encoding="utf-8") as w_file:
                    w_file.write(content)
            except Exception as e:
                logger.error("Failed to write merged markdown file {}: {}", dest_path, e)
        txt_files = list(Path(path).rglob("*.txt"))
        for txt_path in txt_files:
            try:
                with open(txt_path, "r", encoding="utf-8") as r_file:
                    content = r_file.read()
            except Exception as e:
                logger.warning("Failed to read text file {}: {}", txt_path, e)
                continue
            if merge_type == "append":
                for c in captions:
                    image_path = Path(c["id"])
                    if len(image_path.parts) >= 4 and image_path.parts[-4] == txt_path.parts[-3]:
                        content += "\n"
                        content += c["text"].replace("\n\n", "\n") + "\n\n"
            else:
                logger.warning("Text files only support append merge_type, skipping: {}", txt_path)
                continue
            dest_path = os.path.join(self._output_folder, txt_path.name)
            try:
                with open(dest_path, "w", encoding="utf-8") as w_file:
                    w_file.write(content)
            except Exception as e:
                logger.error("Failed to write merged text file {}: {}", dest_path, e)

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
