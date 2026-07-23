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

MULTIMODAL_INIT_PROMPT_CN = (
    "请提供此图像的详细视觉描述。包括关键物体、它们的空间关系、显著的视觉特征以及任何可观察到的动作或事件。"
    "如果存在表格信息，则提取表格的所有信息以及表格间可能存在的逻辑关系，如果没有则不要增加无用内容。"
    "请以清晰、结构化的中文段落进行回答。不要出现英文段落。"
)

MULTIMODAL_FINE_PROMPT_CN = (
    "请根据以上的描述信息增加关于此图像的一些细节信息，如果没有则不要增加无用内容。"
    "请以清晰、结构化的中文段落进行回答。不要出现英文段落。"
)

MULTIMODAL_INIT_PROMPT_EN = (
    "Please provide a detailed visual description of this image. "
    "Include key objects, their spatial relationships, notable visual features, "
    "and any observable actions or events. "
    "Respond in clear, structured English paragraphs."
)

MULTIMODAL_FINE_PROMPT_EN = (
    "Please verify whether the provided text information matches the image content "
    "without any discrepancies. If there are inconsistencies, provide a description "
    "of the necessary corrections and improvements. "
    "Respond in clear, structured English paragraphs."
)
