#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from paddle.base import libpaddle  # noqa: F401
from mx_rag.utils.file_check import SecDirCheck


SUPPORTED_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "PNG"]
MAX_FILE_SIZE_10M = 10 * 1024 * 1024


@dataclass
class MultimodalConfig:
    parser_server: str
    vlm_servers: List[List[str]]
    vlm_model_name: str
    vlm_headers: Dict[str, str] = field(default_factory=lambda: {"Content-Type": "application/json"})
    emb_server_url: Optional[str] = None
    emb_model_name: Optional[str] = None
    filter_type: int = 4
    truncate_dim: int = 512
    num_workers_per_server: int = 8
    batch_size: int = 64
    output_folder: Optional[str] = None
    merge_type: str = "replace"
    prompt_path: Optional[str] = None
    vlm_result_name: str = "tmp"
    device: str = "npu:0"
    timeout: int = 300

    def __post_init__(self):
        if not self.output_folder:
            self.output_folder = os.path.join(os.path.dirname(__file__), "multimodal_output")
            os.makedirs(self.output_folder, 0o750, exist_ok=True)
        SecDirCheck(self.output_folder, MAX_FILE_SIZE_10M).check()
        if not self.parser_server:
            raise ValueError("parser_server must be a non-empty str")
        if not self.vlm_servers:
            raise ValueError("vlm_servers must be configured for service modes")
        if not self.vlm_model_name:
            raise ValueError("vlm_model_name must be a non-empty str")
        if self.merge_type not in ("replace", "append"):
            raise ValueError("merge_type must be 'replace' or 'append'")
        if self.filter_type not in (1, 2, 3, 4):
            raise ValueError("filter_type must be 1, 2, 3, or 4")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.num_workers_per_server < 1:
            raise ValueError("num_workers_per_server must be >= 1")
