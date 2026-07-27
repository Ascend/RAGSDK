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
import multiprocessing
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Tuple

import requests
from loguru import logger
from tqdm import tqdm

from paddle.base import libpaddle  # noqa: F401
from mx_rag.graphrag.multimodal.multimodal_config import MultimodalConfig


class DocumentParser:
    def __init__(self, config: MultimodalConfig):
        self._config = config
        self._batch_size = config.batch_size
        self._K = config.num_workers_per_server
        self.parser_server = config.parser_server
        self._output_folder = config.output_folder
        os.makedirs(self._output_folder, exist_ok=True)
        logger.info("DocumentParser SERVERS: {}", str(self.parser_server))

    def parse(self, file_list: List[str]) -> Tuple[List[str], List[str]]:
        valid_files = self._validate_files(file_list)
        if not valid_files:
            logger.warning("No valid files to parse")
            return [], []
        logger.info("Parsing {} files with Mineru", len(valid_files))
        self._batch_infer(valid_files)
        md_files = self._collect_markdown_files()
        image_files = self._collect_image_files()
        logger.info("Parsed {} markdown files, {} image files", len(md_files), len(image_files))
        return md_files, image_files

    def _validate_files(self, file_list: List[str]) -> List[str]:
        valid = []
        for f in file_list:
            if not os.path.isfile(f):
                logger.warning("File not found: {}", f)
                continue
            valid.append(f)
        return valid

    def _collect_markdown_files(self) -> List[str]:
        output_path = Path(self._output_folder)
        return [str(p) for p in output_path.rglob("*.md")]

    def _collect_image_files(self) -> List[str]:
        output_path = Path(self._output_folder)
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.PNG"]:
            images.extend([str(p) for p in output_path.rglob(ext)])
        return images

    def _progress_update(self, queue, total_tasks: int):
        with tqdm(total=total_tasks, desc="Mineru parsing tasks") as pbar:
            completed = 0
            while completed < total_tasks:
                queue.get()
                completed += 1
                pbar.update(1)

    def _partition_data(self, inputs: List[str]) -> List[List[str]]:
        L = len(inputs)
        if self._K == 0:
            return []
        server_load = math.ceil(L / self._K)
        partitioned_data = []
        start = 0
        while start < L:
            partitioned_data.append(inputs[start : start + server_load])
            start += server_load
        while len(partitioned_data) < self._K:
            partitioned_data.append([])
        return partitioned_data

    def _server_inference(self, server: str, partition: List[str], progress_queue):
        base_url = server.rstrip("/")
        for input_file in partition:
            task_id = None
            try:
                # Step 1: submit parse task via POST /tasks
                with open(input_file, "rb") as f:
                    files = {"files": (os.path.basename(input_file), f)}
                    data = {
                        "backend": "vlm-engine",
                        "return_md": "true",
                        "return_images": "true",
                        "response_format_zip": "true",
                    }
                    resp = requests.post(
                        f"{base_url}/tasks",
                        files=files,
                        data=data,
                        timeout=self._config.timeout,
                    )
                    resp.raise_for_status()
                    submit_result = resp.json()
                    task_id = submit_result.get("task_id")
                    if not task_id:
                        logger.error("No task_id returned for {}", input_file)
                        progress_queue.put(1)
                        continue
                logger.debug("Task {} submitted for {}", task_id, input_file)

                # Step 2: poll GET /tasks/{task_id} until completed or failed
                last_status = None
                start_time = time.time()
                while True:
                    if time.time() - start_time > self._config.timeout:
                        logger.error("Task {} timed out for {}", task_id, input_file)
                        break
                    resp = requests.get(
                        f"{base_url}/tasks/{task_id}",
                        timeout=self._config.timeout,
                    )
                    resp.raise_for_status()
                    task_status = resp.json()
                    current_status = task_status.get("status")
                    if current_status != last_status:
                        logger.debug("Task {} status: {}", task_id, current_status)
                        last_status = current_status
                    if current_status == "completed":
                        break
                    if current_status == "failed":
                        logger.error(
                            "Task {} failed for {}: {}",
                            task_id,
                            input_file,
                            task_status.get("error", "unknown error"),
                        )
                        break
                    time.sleep(2)
                else:
                    continue

                # Step 3: download result ZIP from GET /tasks/{task_id}/result
                resp = requests.get(
                    f"{base_url}/tasks/{task_id}/result",
                    timeout=self._config.timeout,
                )
                resp.raise_for_status()

                # Step 4: extract ZIP to output folder
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                    tmp.write(resp.content)
                    tmp_path = tmp.name
                try:
                    with zipfile.ZipFile(tmp_path, "r") as zf:
                        for member in zf.namelist():
                            member_path = os.path.realpath(os.path.join(self._output_folder, member))
                            if not member_path.startswith(os.path.realpath(self._output_folder)):
                                raise ValueError(f"Zip entry '{member}' attempts path traversal")
                        zf.extractall(self._output_folder)
                finally:
                    os.unlink(tmp_path)
                logger.info("Task {} completed, result extracted for {}", task_id, input_file)

            except requests.RequestException as e:
                logger.error("HTTP error for {} (task_id={}): {}", input_file, task_id, e)
            except Exception as e:
                logger.error("Unexpected error for {} (task_id={}): {}", input_file, task_id, e)

            progress_queue.put(1)

    def _batch_infer(self, inputs: List[str]):
        partitioned_data = self._partition_data(inputs)
        progress_queue = multiprocessing.Manager().Queue()
        progress = multiprocessing.Process(target=self._progress_update, args=(progress_queue, len(inputs)))
        progress.start()
        with multiprocessing.Pool(processes=self._K) as pool:
            for i in range(self._K):
                pool.apply_async(
                    self._server_inference,
                    args=(self.parser_server, partitioned_data[i], progress_queue),
                )
            pool.close()
            pool.join()
        progress.join()

    @staticmethod
    def _batch_generator(lst, batch_size: int):
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]
