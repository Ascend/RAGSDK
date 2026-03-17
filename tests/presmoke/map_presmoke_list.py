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

import os

from loguru import logger

change_st_mapping = {
    "mx_rag/cache": "tests/presmoke/cache",
    "mx_rag/cache/cache_generate_qas/html_makrdown_parser.py": "tests/presmoke/cache/test_markdown_parser.py",
    "mx_rag/cache/cache_generate_qas/generate_qas.py": "tests/presmoke/cache/test_qa_generate.py",
    "mx_rag/chain/img_to_img.py": "tests/presmoke/llm_chain/test_img2img_chain.py",
    "mx_rag/chain/parallel_text_to_text.py": "tests/presmoke/llm_chain/test_parallel_text2text_chain.py",
    "mx_rag/chain/single_text_to_text.py": ["tests/presmoke/knowledge/test_ragsdk_demo.py",
                                            "tests/presmoke/graph/test_graph_pipline.py"],
    "mx_rag/chain/text_to_img.py": "tests/presmoke/llm_chain/test_text2img_chain.py",
    "mx_rag/document/loader/excel_loader.py": "tests/presmoke/document/test_presmoke_excel_loader.py",
    "mx_rag/embedding": "tests/presmoke/embedding/test_embedding_factory.py",
    "mx_rag/embedding/local/sparse_embedding.py": "tests/presmoke/embedding/test_presmoke_sparse_embedding.py",
    "mx_rag/graphrag": "tests/presmoke/graph",
    "mx_rag/knowledge": "tests/presmoke/knowledge",
    "mx_rag/llm/img2img.py": "tests/presmoke/llm_chain/test_img2img_chain.py",
    "mx_rag/llm/text2img.py": "tests/presmoke/llm_chain/test_text2img_chain.py",
    "mx_rag/llm/text2text.py": "tests/presmoke/knowledge/test_ragsdk_demo.py",
    "mx_rag/retrievers/bm_retriever.py": "tests/presmoke/retrieval/test_bm_retriever.py",
    "mx_rag/summary": "tests/presmoke/summary"
}

with open("changed_files.txt", "r") as fh:
    lines = fh.readlines()

presmoke_list = []
for line in lines:
    line = line.strip()
    # line在change_st_mapping中按照最长匹配原则，返回presmoke用例列表并去重

    mapped_key = ""
    # 最长匹配
    for key in change_st_mapping.keys():
        if line.startswith(key) and len(key) > len(mapped_key):
            mapped_key = key
    if mapped_key:
        presmoke_file_path = change_st_mapping[mapped_key]
        if not os.path.exists(presmoke_file_path):
            continue
        if isinstance(presmoke_file_path, list):
            presmoke_list.extend(presmoke_file_path)
        else:
            presmoke_list.append(presmoke_file_path)
    # 默认跑问答demo
    else:
        presmoke_list.append("tests/presmoke/knowledge/test_ragsdk_demo.py")

presmoke_list = list[str](set[str](presmoke_list))
logger.info(f"presmoke_list: {presmoke_list}")

with open("map_presmoke_list.txt", "w") as fh:
    for item in presmoke_list:
        fh.write(item + "\n")
