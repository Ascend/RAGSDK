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
import json
import os
import random
import traceback
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from loguru import logger
from datasets import load_dataset

from mx_rag.corag.config import CoRagBaseConfig, DEFAULT_TASK_DESC, DEFAULT_SAMPLE_TASK_DESC
from mx_rag.corag.corag_agent import CoRagAgent
from mx_rag.corag.utils import check_answer, check_answer_with_llm_judge
from mx_rag.utils import file_check
from mx_rag.utils.common import MAX_FILE_SIZE, validate_params
from mx_rag.corag.prompts import (get_generate_subquery_prompt, get_generate_intermediate_answer_prompt,
                                  get_generate_final_answer_prompt)

# Constants
DEFAULT_RANDOM_ID_RANGE = 1_000_000
MAX_ERROR_LOGS = 5


class SampleGenerator:
    """样本生成器类，负责生成CoRAG样本
    
    该类通过多线程并行处理输入数据，为每个查询生成有效的推理路径，
    并将其转换为可用于训练的样本格式。
    """
    @validate_params(config=dict(validator=lambda x: isinstance(x, CoRagBaseConfig), 
                                 message="config must be a CoRagBaseConfig instance"))
    def __init__(self, config: CoRagBaseConfig):
        self.config = config
        self.base_llm = config.base_llm
        self.final_llm = config.final_llm
        self.sub_answer_llm = config.sub_answer_llm
        self.judge_llm = config.judge_llm
        self.retrieve_api_url = config.retrieve_api_url
        self.num_threads = config.num_threads
        self.max_path_length = config.max_path_length

    def _create_agent(self) -> CoRagAgent:
        """为每个线程创建独立的 Agent 实例，避免线程安全问题。
        
        Returns:
            新创建的 CoRagAgent 实例。
        """
        return CoRagAgent(
            base_llm=self.base_llm,
            retrieve_api_url=self.retrieve_api_url,
            final_llm=self.final_llm,
            sub_answer_llm=self.sub_answer_llm,
        )

    @staticmethod
    def _generate_samples(path: Dict) -> List[Dict[str, Any]]:
        """根据有效路径生成对齐的训练样本。
        
        Args:
            path: 包含查询、步骤和最终答案的有效路径字典。
            
        Returns:
            对齐的训练样本列表，每个样本包含类型和消息列表。
        """
        aligned_samples = []
        # 1. 提取核心字段
        main_query = path.get('query', '')
        steps = path.get('steps', [])
        final_answer = path.get('generated_final_answer') or (
            path.get('answers', [""])[0] if path.get('answers') else "")

        if not main_query or not steps:
            return []

        history_sq: List[str] = []
        history_sa: List[str] = []

        for step in steps:
            subquery = step.get('subquery', '').strip()
            subanswer = step.get('subanswer', '').strip()
            docs = step.get('documents', '')

            if not subquery or not subanswer:
                continue

            prompt_sq = get_generate_subquery_prompt(main_query, history_sq, history_sa, DEFAULT_SAMPLE_TASK_DESC)
            aligned_samples.append({
                "type": "subquery_generation",
                "messages": [{'role': 'user', 'content': prompt_sq}] +
                            [{"role": "assistant", "content": f"SubQuery: {subquery}"}]
            })

            prompt_sa = get_generate_intermediate_answer_prompt(subquery, docs)
            aligned_samples.append({
                "type": "subanswer_generation",
                "messages": [{'role': 'user', 'content': prompt_sa}] +
                            [{"role": "assistant", "content": f"SubAnswer: {subanswer}"}]
            })

            history_sq.append(subquery)
            history_sa.append(subanswer)

        if final_answer:
            prompt_final = get_generate_final_answer_prompt(
                main_query, 
                history_sq, 
                history_sa, 
                DEFAULT_SAMPLE_TASK_DESC
            )
            aligned_samples.append({
                "type": "final_answer_generation",
                "messages": [
                    {'role': 'user', 'content': prompt_final}, 
                    {"role": "assistant", "content": f"Final Answer: {final_answer}"}
                ]
            })

        return aligned_samples

    def _generate_paths(self, item: Dict, agent: CoRagAgent, n_samples: int) -> List[Dict]:
        """为单个查询生成多个推理路径，返回有效路径。
        
        Args:
            item: 包含查询和答案的字典。
            agent: CoRagAgent实例，用于路径采样和答案生成。
            
        Returns:
            有效路径列表，每个路径包含查询、步骤和生成的最终答案。
        """
        query = item.get('query', '')
        ground_truths = item.get('answer', '')
        if not query:
            logger.warning("Empty query encountered, skipping")
            return []

        valid_paths = []
        for sample_idx in range(n_samples):
            try:
                path = agent.sample_path(
                    query=query,
                    task_desc=DEFAULT_TASK_DESC,
                    max_path_length=self.max_path_length,
                )

                # Generate final answer based on the path
                final_ans = agent.generate_final_answer(
                    rag_path=path,
                    task_description=DEFAULT_SAMPLE_TASK_DESC,
                )
                # Check correctness
                is_correct = (
                    check_answer_with_llm_judge(final_ans, ground_truths, query, self.judge_llm)
                    if self.judge_llm
                    else check_answer(final_ans, ground_truths)
                )

                if is_correct:
                    valid_path = {
                        "id": item.get('id', str(random.randint(0, DEFAULT_RANDOM_ID_RANGE))),
                        "query": query,
                        "answers": ground_truths,
                        "generated_final_answer": final_ans,
                        "steps": [
                            {
                                "subquery": path.subqueries[i],
                                "subanswer": path.subanswers[i],
                                "documents": path.documents[i]
                            }
                            for i in range(len(path.subqueries))
                        ]
                    }
                    valid_paths.append(valid_path)
                    break
            except Exception as e:
                logger.warning(f"Path generation attempt {sample_idx + 1} failed: {e}")
                continue
        return valid_paths

    def _process_item(self, item: Dict, n_samples: int) -> List[Dict]:
        """处理单个数据项，生成对齐样本。
        
        Args:
            item: 输入数据项，包含查询和答案。
            
        Returns:
            生成的训练样本列表，如果生成失败则返回空列表。
        """
        agent = self._create_agent()
        valid_paths = self._generate_paths(item, agent, n_samples)
        if not valid_paths:
            return []
        return self._generate_samples(valid_paths[0])


    def generate(self, input_file: str, output_file: str, n_samples: int = 5):
        """生成样本主方法。

        从输入文件加载数据，并行处理生成训练样本，并保存到输出文件。
        
        Args:
            input_file: 输入数据文件路径（JSON格式）。
            output_file: 输出文件路径。
            n_samples: 每个查询采样的路径数量，默认为5。
            
        Returns:
            处理后的样本列表。
            
        Raises:
            ValueError: 当输入文件路径为空时抛出。
        """
        if not input_file:
            raise ValueError("input_file cannot be empty")
        file_check.SecFileCheck(input_file, MAX_FILE_SIZE).check()
        logger.info("Loading input data...")
        input_data = load_dataset("json", data_files=input_file)['train']

        total_cnt = len(input_data)
        results_map: Dict[int, List[Dict]] = {}
        error_count = 0

        logger.info(f"Processing {total_cnt} items with {self.num_threads} threads...")

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_index = {
                executor.submit(self._process_item, item, n_samples): i 
                for i, item in enumerate(input_data)
            }

            for future in tqdm(as_completed(future_to_index), total=total_cnt, desc="Processing"):
                index = future_to_index[future]
                try:
                    res = future.result()
                    if res:
                        results_map[index] = res
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing item at index {index}: {e}")
                    if error_count <= MAX_ERROR_LOGS:
                        traceback.print_exc()

        processed_results = [results_map[i] for i in range(len(input_data)) if i in results_map]

        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                for items in processed_results:
                    for item in items:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"Results saved to {os.path.basename(output_file)}")

        return processed_results
