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
import json
import time
import re
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from loguru import logger

from mx_rag.corag.prompts import get_generate_intermediate_answer_prompt
from mx_rag.corag.corag_agent import CoRagAgent
from mx_rag.corag.utils import search_by_retrieve_api, normalize_text
from mx_rag.corag.utils import ThreadSafeCounter, check_answer, check_answer_with_llm_judge
from mx_rag.corag.config import CoRagBaseConfig, DEFAULT_SAMPLE_TASK_DESC, DEFAULT_TASK_DESC
from mx_rag.utils import file_check
from mx_rag.utils.common import MAX_FILE_SIZE


class CoRagEvaluator:
    """CoRAG评估器类
    
    该类通过多线程并行处理评估数据，计算检索召回率等指标，
    并生成详细的评估报告。
    """

    def __init__(self, config: CoRagBaseConfig):
        """初始化评估器。
        
        Args:
            config: 配置对象，包含LLM实例、API地址和并行参数等。
        """
        self.config = config
        self.base_llm = config.base_llm
        self.final_llm = config.final_llm
        self.judge_llm = config.judge_llm
        self.sub_answer_llm = config.sub_answer_llm
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
    def _unicode_escape_to_char(s: str) -> str:
        """将含 Unicode 转义的字符串还原为原始字符"""
        try:
            return s.encode('utf-8').decode('unicode_escape')
        except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
            # 如果转换失败，返回原始字符串
            return s

    def _check_hit(self, retrieved_docs: List[str], golden_facts: List[str]) -> Tuple[int, int]:
        """计算基于"Soft Inclusion"原则的命中次数。
        
        命中条件：
        1. Golden chunk 是 Retrieved chunk 的子串
        2. Retrieved chunk 是 Golden chunk 的子串且长度 > 0.5 * Golden 长度
        
        Args:
            retrieved_docs: 检索到的文档列表。
            golden_facts: 黄金事实列表。
            
        Returns:
            元组 (命中次数, 黄金事实总数)。
        """
        if not golden_facts:
            return 0, 0

        norm_gold = [normalize_text(g) for g in golden_facts]
        norm_retr = [normalize_text(self._unicode_escape_to_char(r)) for r in retrieved_docs]

        hits = 0
        for g in norm_gold:
            if not g:
                continue
            is_hit = any(
                g in r or (r in g and len(r) > 0.5 * len(g))
                for r in norm_retr if r
            )
            if is_hit:
                hits += 1

        return hits, len(golden_facts)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """按标点符号分割文本为句子。
        
        Args:
            text: 待分割的文本。
            
        Returns:
            句子列表。
        """
        if text is None:
            return []
        # 限制输入长度，防止ReDoS攻击
        if len(text) > 10000:
            return [text.strip()]
        return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]

    def _get_golden_facts(self, item: Dict[str, Any]) -> List[str]:
        """从数据项中提取黄金事实。
        
        支持格式：
        1. MuSiQue: "paragraphs" 列表，包含 "is_supporting" 标志
        2. HotpotQA: "context" 和 "supporting_facts" 字段
        
        Args:
            item: 数据项字典。
            
        Returns:
            黄金事实句子列表。
        """
        if 'paragraphs' in item:
            facts = []
            for p in item['paragraphs']:
                if p.get('is_supporting'):
                    text = p.get('paragraph_text', '')
                    if text:
                        facts.extend(self._split_sentences(text))
            return facts

        context = item.get('context', [])
        supporting_facts = item.get('supporting_facts', [])
        ctx_map = {c[0]: c[1] for c in context}

        facts = []
        for title, sent_idx in supporting_facts:
            if title in ctx_map:
                sentences = ctx_map[title]
                if 0 <= sent_idx < len(sentences):
                    facts.append(sentences[sent_idx])

        return facts

    @staticmethod
    def _extract_question(item: Dict[str, Any]) -> str:
        """从数据项中提取问题文本。
        
        Args:
            item: 数据项字典。
            
        Returns:
            问题文本，如果未找到则返回空字符串。
        """
        question = item.get('question', '')
        if not question:
            for k in item.keys():
                if 'question' in k.lower():
                    question = item[k]
                    break
        return question

    def _naive_retrieve(self, query: str, num_contexts: int) -> List[str]:
        """执行朴素检索。
        
        Args:
            query: 查询文本。
            num_contexts: 检索上下文数量。
            
        Returns:
            检索到的文档列表。
        """
        naive_docs = []
        try:
            retriever_results = search_by_retrieve_api(query=query, url=self.retrieve_api_url)
            for res in retriever_results:
                if isinstance(res, str):
                    naive_docs.append(res)
                elif isinstance(res, dict):
                    content = res.get('contents') or res.get('content') or res.get('text') or str(res)
                    naive_docs.append(content)
        except Exception as e:
            logger.warning(f"Naive retrieval failed: {e}")

        return naive_docs[:num_contexts]
    
    def _generate_naive_prediction(self, question: str, naive_docs: List[str]) -> str:
        prompt = get_generate_intermediate_answer_prompt(question, naive_docs)
        prediction = self.base_llm.chat(prompt)
        return prediction.strip()
    
    def _select_documents_for_recall(self, documents: List[List[str]], num_contexts: int) -> List[str]:
        """智能选择文档用于召回率计算，确保文档来源的多样性和公平性。
        
        Args:
            documents: 所有子查询的文档列表，每个子查询对应一个文档列表
            num_contexts: 文档数量限制
            
        Returns:
            选择后的文档列表
        """
        all_documents = []
        document_sources = []  # Track which subquery each document comes from
        
        if documents:
            for subquery_idx, docs in enumerate(documents):
                for doc in docs:
                    all_documents.append(doc)
                    document_sources.append(subquery_idx)
        
        # Remove duplicate documents while preserving order and source information
        seen = set()
        unique_documents = []
        unique_sources = []
        
        for doc, source in zip(all_documents, document_sources):
            if doc not in seen:
                seen.add(doc)
                unique_documents.append(doc)
                unique_sources.append(source)
        
        # Apply a sophisticated document selection strategy
        if unique_documents:
            num_subqueries = max(unique_sources) + 1 if unique_sources else 1
            # Calculate base allocation per subquery
            base_per_subquery = num_contexts // num_subqueries
            # Calculate remaining documents after base allocation
            remaining = num_contexts % num_subqueries
            
            # Allocate documents from each subquery
            limited_documents = []
            for subquery_idx in range(num_subqueries):
                # Select documents from current subquery
                subquery_docs = [doc for doc, src in zip(unique_documents, unique_sources) if src == subquery_idx]
                # Determine how many documents to take from this subquery
                take_count = base_per_subquery + (1 if subquery_idx < remaining else 0)
                # Add documents to the result
                limited_documents.extend(subquery_docs[:take_count])
                # If we've already reached the limit, break
                if len(limited_documents) >= num_contexts:
                    break
            
            # If still not enough documents, fill with remaining documents
            if len(limited_documents) < num_contexts:
                # Get documents not yet selected
                selected_docs = set(limited_documents)
                remaining_docs = [doc for doc in unique_documents if doc not in selected_docs]
                # Add remaining documents up to the limit
                limited_documents.extend(remaining_docs[:num_contexts - len(limited_documents)])
        else:
            limited_documents = []
        
        return limited_documents

    def _process_item(
        self,
        item: Dict[str, Any],
        agent: CoRagAgent,
        num_contexts: int,
        calc_recall: bool,
        enable_naive_retrieval: bool,
    ) -> Optional[Dict[str, Any]]:
        """处理单个评估数据项。
        
        Args:
            item: 数据项字典，包含问题和答案。
            agent: CoRagAgent实例。
            num_contexts: 检索上下文数量。
            calc_recall: 是否计算召回率。
            enable_naive_retrieval: 是否启用朴素检索对比。
            
        Returns:
            评估结果字典，如果处理失败则返回None。
        """
        question = self._extract_question(item)
        if not question:
            logger.warning("Skipping item without question")
            return None

        ground_truth = item.get('answer', '')
        start_time = time.time()
        
        naive_prediction = None
        is_naive_correct = None

        path = agent.sample_path(
            query=question,
            task_desc=DEFAULT_TASK_DESC,
            max_path_length=self.max_path_length,
        )
        path_gen_time = time.time() - start_time

        corag_recall_info = {}
        naive_recall_info = {}

        if calc_recall:
            golden_facts = self._get_golden_facts(item)
            
            # Select documents using the new intelligent strategy
            selected_documents = self._select_documents_for_recall(path.documents, num_contexts)

            c_hits, c_total = self._check_hit(selected_documents, golden_facts)
            corag_recall_info = {
                "hits": c_hits,
                "total": c_total,
                "recall": c_hits / c_total if c_total > 0 else 0.0
            }

            if enable_naive_retrieval and self.retrieve_api_url:
                naive_docs = self._naive_retrieve(question, num_contexts)
                n_hits, n_total = self._check_hit(naive_docs, golden_facts)
                naive_recall_info = {
                    "hits": n_hits,
                    "total": n_total,
                    "recall": n_hits / n_total if n_total > 0 else 0.0,
                    }
                naive_prediction = self._generate_naive_prediction(question, naive_docs)
                is_naive_correct = ( 
                    check_answer_with_llm_judge(naive_prediction, [ground_truth], question, self.judge_llm)
                    if self.judge_llm
                    else check_answer(naive_prediction, [ground_truth])
                    )

        prediction = agent.generate_final_answer(
            rag_path=path,
            task_description=DEFAULT_SAMPLE_TASK_DESC,
        )

        end_time = time.time()
        total_time = end_time - start_time
        final_gen_time = total_time - path_gen_time
        is_correct = (
            check_answer_with_llm_judge(prediction, [ground_truth], question, self.judge_llm)
            if self.judge_llm
            else check_answer(prediction, [ground_truth])
        )
        return {
            "question": question,
            "ground_truth": ground_truth,
            "corag_prediction": prediction,
            "naive_prediction": naive_prediction if enable_naive_retrieval else None,
            "is_correct": is_correct,
            "naive_is_correct": is_naive_correct if enable_naive_retrieval else None,
            "reasoning_steps": [
                {"subquery": sq, "subanswer": sa}
                for sq, sa in zip(path.subqueries or [], path.subanswers or [])
            ],
            "time": [path_gen_time, final_gen_time],
            "corag_recall": corag_recall_info if calc_recall else None,
            "naive_recall": naive_recall_info if calc_recall and enable_naive_retrieval else None
        }

    @staticmethod
    def _aggregate_metrics(
        results: List[Dict[str, Any]],
        calc_recall: bool,
        enable_naive_retrieval: bool,
    ) -> Dict[str, Any]:
        """聚合评估指标。
        
        Args:
            results: 评估结果列表。
            calc_recall: 是否计算召回率。
            enable_naive_retrieval: 是否启用朴素检索对比。
            
        Returns:
            聚合后的指标字典。
        """
        total_path_time = total_time = 0.0
        total_corag_hits = total_naive_hits = total_gold_chunks = 0
        corag_total_correct, naive_total_correct = 0, 0
        for res in results:
            t = res.get("time", [0, 0])
            total_path_time += t[0]
            total_time += t[1]
            if res.get("is_correct"):
                corag_total_correct += 1
            if res.get("naive_is_correct"):
                naive_total_correct += 1
            if calc_recall and res.get("corag_recall"):
                total_corag_hits += res["corag_recall"]["hits"]
                total_gold_chunks += res["corag_recall"]["total"]

            if calc_recall and enable_naive_retrieval and res.get("naive_recall"):
                total_naive_hits += res["naive_recall"]["hits"]

        num_samples = len(results)
        summary = {
            "type": "Summary",
            "total_samples": num_samples,
            "corag_accuracy": corag_total_correct / num_samples if num_samples else 0.0,
            "naive_accuracy": naive_total_correct / num_samples if num_samples else 0.0,
            "corag_correct_count": corag_total_correct,
            "naive_correct_count": naive_total_correct,
            "avg_path_time": total_path_time / num_samples if num_samples else 0,
            "avg_time": total_time / num_samples if num_samples else 0,
        }

        if calc_recall and total_gold_chunks > 0:
            summary["corag_micro_recall"] = total_corag_hits / total_gold_chunks
            if enable_naive_retrieval:
                summary["naive_micro_recall"] = total_naive_hits / total_gold_chunks

        return summary

    def evaluate(
        self,
        eval_file: str,
        save_file: str,
        calc_recall: bool = True,
        enable_naive_retrieval: bool = True,
        num_contexts: int = 10
    ) -> List[Dict[str, Any]]:
        """执行评估主方法。
        
        从评估文件加载数据，并行处理生成评估结果，并保存到输出文件。
        
        Args:
            eval_file: 评估数据文件路径（JSON格式）。
            save_file: 结果保存文件路径。
            calc_recall: 是否计算召回率，默认为True。
            enable_naive_retrieval: 是否启用朴素检索对比，默认为True。
            num_contexts: 检索上下文数量，默认为10。
            
        Returns:
            评估结果列表，第一个元素是聚合指标。
            
        Raises:
            ValueError: 当评估文件路径为空时抛出。
        """
        if not eval_file:
            raise ValueError("eval_file cannot be empty")
        file_check.SecFileCheck(eval_file, MAX_FILE_SIZE).check()
        logger.info(f"Loading custom dataset from {os.path.basename(eval_file)}...")
        with open(eval_file, 'r', encoding='utf-8') as f:
            data_items = json.load(f)

        total_cnt = len(data_items)
        results_map: Dict[int, Dict[str, Any]] = {}
        error_count = 0
        processed_cnt = ThreadSafeCounter()

        logger.info(f"Processing {total_cnt} items with {self.num_threads} threads...")

        def process_wrapper(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            agent = self._create_agent()
            result = self._process_item(
                item, agent, num_contexts, calc_recall, enable_naive_retrieval
            )
            cnt = processed_cnt.increment()
            if cnt % 10 == 0:
                logger.info(f"Processed {cnt} / {total_cnt}")
            return result

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_index = {
                executor.submit(process_wrapper, item): i
                for i, item in enumerate(data_items)
            }

            for future in tqdm(as_completed(future_to_index), total=total_cnt, desc="Evaluating"):
                index = future_to_index[future]
                try:
                    res = future.result()
                    if res:
                        results_map[index] = res
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing item at index {index}")
                    if error_count <= 5:
                        import traceback
                        traceback.print_exc()

        # 使用更清晰的方式处理结果，保持顺序且避免索引问题
        processed_results = []
        for i in range(len(data_items)):
            if i in results_map:
                processed_results.append(results_map[i])

        summary = self._aggregate_metrics(processed_results, calc_recall, enable_naive_retrieval)
        processed_results.insert(0, summary)

        if save_file:
            output_dir = os.path.dirname(save_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(processed_results, f, ensure_ascii=False, indent=4)
            logger.info(f"Results saved to {os.path.basename(save_file)}")

        # 添加边界检查，避免当processed_results为空时出现-1
        num_processed = len(processed_results) - 1 if len(processed_results) > 0 else 0
        logger.info(f"Evaluated {num_processed}/{total_cnt} items, {error_count} errors")
        logger.info(
            f"Accuracy: {summary.get('corag_accuracy', 'N/A')} "
            f"({summary.get('corag_correct_count', 0)}/{num_processed})"
        )
        logger.info(
            f"Naive Accuracy: {summary.get('naive_accuracy', 'N/A')} "
            f"({summary.get('naive_correct_count', 0)}/{num_processed})"
        )
        logger.info(f"CoRAG Micro Recall: {summary.get('corag_micro_recall', 'N/A')}")
        if enable_naive_retrieval:
            logger.info(f"Naive Micro Recall: {summary.get('naive_micro_recall', 'N/A')}")

        return processed_results