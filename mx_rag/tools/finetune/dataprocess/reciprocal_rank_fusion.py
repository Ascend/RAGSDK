# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

# Reciprocal Rank Fusion(RRF)
# 倒数排序融合（RRF）是一种将具有不同相关性指标的多个结果集组合成单个结果集的方法

MAX_FUSION_LISTS = 10
MAX_LISTS_LEN = 10000


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: float = 60):
    if k <= 0:
        raise Exception(f"k must large than 0, now is: {k}")

    if len(rank_lists) > MAX_FUSION_LISTS:
        raise Exception(f"rank_lists should not large {MAX_FUSION_LISTS}, now is: {len(rank_lists)}")

    for rank_list in rank_lists:
        if len(rank_list) > MAX_LISTS_LEN:
            raise Exception(f"rank_list should not longer than {MAX_LISTS_LEN}, now is: {len(rank_list)}")

    # k是常数平滑因子
    fused_rank = {}

    for rank_list in rank_lists:
        for rank, item in enumerate(rank_list):
            rank_score = 1 / (rank + k)
            if item in fused_rank:
                fused_rank[item] += rank_score
            else:
                fused_rank[item] = rank_score

    # 对最终的融合排名列表进行排序
    def sort_key(key):
        return fused_rank.get(key, 0)

    fused_keys = list(fused_rank.keys())
    fused_keys.sort(reverse=True, key=sort_key)
    return fused_keys
