# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import Optional

from mx_rag.utils.common import validate_params, INT_32_MAX, BOOL_TYPE_CHECK_TIP


class LLMParameterConfig:
    """
    调用大模型相关参数，适配mindie默认值

    Args:
        max_tokens: int = 512, 允许推理生成的最大token个数
        presence_penalty: float = 0.0,影响模型如何根据到目前为止是否出现在文本中来惩罚新token。正值将通过惩罚已经使用的词，增加模型谈论新主题的可能性。
        frequency_penalty: float = 0.0,影响模型如何根据文本中词汇（token）的现有频率惩罚新词汇（token）。正值将通过惩罚已经频繁使用的词来降低模型一行中重复用词的可能性。
        temperature: float = 1.0,控制生成的随机性，较高的值会产生更多样化的输出
        top_p: float = 1.0,控制模型生成过程中考虑的词汇范围，使用累计概率选择候选词，直到累计概率超过给定的阈值。
        seed: Optional[int] = None,用于指定推理过程的随机种子
        stream: bool = False,是否使用流式回答，默认False
    """
    @validate_params(
        max_tokens=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= INT_32_MAX,
                        message="param must be int and value range [1, 2 ** 31 - 1]"),
        presence_penalty=dict(validator=lambda x: isinstance(x, float) and -2.0 <= x <= 2.0,
                              message="param must be float and value range [-2.0, 2.0]"),
        frequency_penalty=dict(validator=lambda x: isinstance(x, float) and -2.0 <= x <= 2.0,
                               message="param must be float and value range [-2.0, 2.0]"),
        temperature=dict(validator=lambda x: isinstance(x, float) and 0.0 < x <= 2.0,
                         message="param must be float and value range (0.0, 2.0]"),
        top_p=dict(validator=lambda x: isinstance(x, float) and 0.0 < x <= 1.0,
                   message="param must be float and value range (0.0, 1.0]"),
        seed=dict(validator=lambda x: x is None or (isinstance(x, int) and 0 < x <= INT_32_MAX),
                  message="param must be None or int, and int value range (0, 2 ** 31 - 1]"),
        stream=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
    )
    def __init__(self, max_tokens: int = 512, presence_penalty: float = 0.0,
                 frequency_penalty: float = 0.0, temperature: float = 1.0,
                 top_p: float = 1.0, seed: Optional[int] = None,
                 stream: bool = False):
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.stream = stream
