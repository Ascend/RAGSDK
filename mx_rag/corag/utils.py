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

import threading
import re
from typing import List, Dict

import requests
from loguru import logger

from mx_rag.corag.prompts import get_evaluate_answer_prompt
from mx_rag.llm.text2text import Text2TextLLM


def _normalize_retrieve_api_results(results):
    """Normalize retrieve API responses into a list for downstream consumers."""
    if isinstance(results, dict):
        for key in ['chunks', 'data', 'results', 'docs', 'passages']:
            value = results.get(key)
            if isinstance(value, list):
                return value
        return [results] if results else []
    if isinstance(results, list):
        return results
    return []


def truncate_long_text_by_char(text: str, max_token_length: int) -> str:
    """
    按字符数截断长文本，确保中文和英文的字符比例符合预期。
    保留文本的开头和结尾部分，避免丢失重要信息。
    Args:
        text: 待截断的文本字符串
        max_token_length: 允许的最大字符长度
        
    Returns:
        截断后的文本字符串
    """
    if not text:
        return text
    chinese_ratio = sum(1 for c in text if '\u4e00' <= c <= '\u9fff') / len(text)
    max_char_len = max_token_length if chinese_ratio > 0.5 else max_token_length * 2
    if len(text) <= max_char_len:
        return text
    half_len = max_char_len // 2
    return text[:half_len] + text[- (max_char_len - half_len):]


def search_by_retrieve_api(query: str, url: str, top_k: int = 5) -> List[Dict]:
    try:
        response = requests.post(url, json={'query': query, 'top_k': top_k}, headers={"Content-Type": "application/json"}, timeout=600)
        if response.status_code == 200:
            return _normalize_retrieve_api_results(response.json())
        else:
            logger.error(f"Failed to get a response from retrieve API. Status code: {response.status_code}")
            return []
    except requests.RequestException as e:
        logger.error(f"Error calling retrieve API: {type(e).__name__}: Connection error occurred")
        return []


def normalize_text(text: str) -> str:
    """标准化文本：小写、移除标点、冠词和多余空格。"""
    if text is None:
        return ""
    # 限制输入长度，防止ReDoS攻击
    if len(text) > 10000:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def check_answer(prediction: str, ground_truths: List[str]) -> bool:
    """检查预测答案是否匹配任一标准答案。     
    Args:
        prediction: 预测的答案。
        ground_truths: 标准答案列表。
        
    Returns:
        如果匹配成功返回True，否则返回False。
    """
    if not prediction:
        return False

    norm_pred = normalize_text(prediction)

    for gt in ground_truths:
        norm_gt = normalize_text(gt)
        if not norm_gt:
            continue

        if norm_pred == norm_gt or norm_gt in norm_pred:
            return True

    return False


def check_answer_with_llm_judge(
        prediction: str,
        ground_truths: List[str],
        query: str,
        judge_llm: Text2TextLLM,
) -> bool:
    """使用LLM作为评判者检查预测答案是否正确。
    
    Args:
        prediction: 预测的答案。
        ground_truths: 标准答案列表。
        query: 原始查询。
        judge_llm: 用于评判的LLM实例。
        
    Returns:
        如果答案正确返回True，否则返回False。
    """
    if not prediction:
        return False

    # Format ground truths
    gt_text = " or ".join([f'"{gt}"' for gt in ground_truths if gt])
    if not gt_text:
        return False

    prompt = get_evaluate_answer_prompt(query, prediction, gt_text)

    try:
        response = judge_llm.chat(query=prompt)
        response_upper = response.strip().upper()
        # Check if response starts with "YES" or contains "YES" before "NO"
        if response_upper.startswith("YES"):
            return True
        # Check if "YES" appears before "NO" in the response
        yes_pos = response_upper.find("YES")
        no_pos = response_upper.find("NO")
        if yes_pos != -1 and (no_pos == -1 or yes_pos < no_pos):
            return True
        return False
    except Exception as e:
        logger.warning(f"LLM judge error: {e}, falling back to string matching")
        return check_answer(prediction, ground_truths)


class ThreadSafeCounter:
    def __init__(self, initial_value=0):
        """Initialize a thread-safe counter with the specified initial value.
        
        Args:
            initial_value: Starting value for the counter (default: 0)
        """
        self._count = initial_value
        self._thread_lock = threading.Lock()

    def increment(self, step=1):
        """Atomically increment the counter by the specified step and return the updated value.
        
        Args:
            step: Value to increment the counter by (default: 1)
            
        Returns:
            The new value of the counter after increment
        """
        with self._thread_lock:
            self._count += step
            return self._count

    def reset(self):
        """Reset the counter to zero and return the new value."""
        with self._thread_lock:
            self._count = 0
            return self._count
