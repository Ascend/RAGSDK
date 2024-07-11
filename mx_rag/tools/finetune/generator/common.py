# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from loguru import logger

from mx_rag.llm import Text2TextLLM
from mx_rag.tools.finetune.dataprocess import bm25_featured, llm_preferred, reranker_featured, reciprocal_rank_fusion


class BaseGenerator:

    def __init__(self, llm: Text2TextLLM):
        self.llm = llm

    @staticmethod
    def _feature_qd_pair(query_list: list[str], doc_list: list[str], reranker: str, featured_percentage: float):
        """文档精选，使用bm25和reranker共同打分，按比率保留前面的问答对"""

        logger.info("Selection-bm25 Scoring")
        bm25_scores = bm25_featured(query_list, doc_list)
        bm25_sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_query_list = [query_list[i] for i in bm25_sorted_indices]

        logger.info("Selection-reranker Scoring")
        reranker_scores = reranker_featured(reranker, query_list, doc_list)
        reranker_sorted_indices = sorted(range(len(reranker_scores)), key=lambda i: reranker_scores[i], reverse=True)
        reranker_query_list = [query_list[i] for i in reranker_sorted_indices]

        logger.info("RRF algorithm fuses two sorting results")
        fused_query_list = reciprocal_rank_fusion([bm25_query_list, reranker_query_list])

        # 将两个列表打包成元组列表
        zipped_lists = list(zip(query_list, doc_list))

        # 自定义排序函数，根据排序列表的顺序对元组进行排序
        def custom_sort(item):
            return fused_query_list.index(item[0])

        # 根据自定义排序函数对元组列表进行排序
        sorted_zipped_lists = sorted(zipped_lists, key=custom_sort)
        # 将排序后的元组列表拆解成两个元组
        sorted_query_list, sorted_doc_list = zip(*sorted_zipped_lists)

        logger.info(f"Select the top {featured_percentage * 100}% data as the featured set based "
                    f"on the set parameters")
        featured_query_list = list(sorted_query_list[:round(len(sorted_query_list) * featured_percentage)])
        featured_doc_list = list(sorted_doc_list[:round(len(sorted_doc_list) * featured_percentage)])

        return featured_query_list, featured_doc_list

    def _prefer_qd_pair(self, featured_query_list: list[str], featured_doc_list: list[str], llm_threshold_score: float):
        """大模型精选"""

        logger.info("LLM score and eliminate those whose scores are lower than the preset threshold")
        llm_scores = llm_preferred(self.llm, featured_query_list, featured_doc_list)
        # 使用列表推导式筛选出所有低于阈值分数的数据，并统计筛选结果的长度
        count_upper_threshold_score = len([x for x in llm_scores if x >= llm_threshold_score])

        llm_sorted_indices = sorted(range(len(llm_scores)), key=lambda i: llm_scores[i], reverse=True)
        llm_query_list = [featured_query_list[i] for i in llm_sorted_indices]
        llm_doc_list = [featured_doc_list[i] for i in llm_sorted_indices]

        preferred_query_list = llm_query_list[:count_upper_threshold_score]
        preferred_doc_list = llm_doc_list[:count_upper_threshold_score]

        return preferred_query_list, preferred_doc_list
