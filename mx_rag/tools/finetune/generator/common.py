# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
from pathlib import Path
from typing import Callable, List

from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from mx_rag.document.loader import DocxLoader
from mx_rag.utils.common import validata_list_str, TEXT_MAX_LEN, STR_MAX_LEN
from mx_rag.utils.file_check import FileCheck, SecFileCheck
from mx_rag.llm import Text2TextLLM
from mx_rag.reranker.local import LocalReranker
from mx_rag.tools.finetune.dataprocess \
    import bm25_featured, generate_qa_embedding_pairs, llm_preferred, reranker_featured, reciprocal_rank_fusion
from mx_rag.utils.file_operate import read_jsonl_from_file, write_jsonl_to_file

MAX_DATASET_LEN = 10000
MAX_FILE_SIZE_100M = 100 * 1024 * 1024
MAX_FILE_PROCESS_TIMES = 1000
SAMPLE_RANGE_MIN = 100


class BaseGenerator:

    def __init__(self, llm: Text2TextLLM, dataset_path: str):
        self.llm = llm
        self.dataset_path = dataset_path

    @staticmethod
    def generate_origin_document(document_path: str, filter_func: Callable[[List[str]], List[str]] = None,
                                 chunk_size: int = 512, chunk_overlap: int = 10):
        logger.info("Original document splitting")
        FileCheck.dir_check(document_path)

        def doc_load(file_path: str):
            SecFileCheck(file_path, MAX_FILE_SIZE_100M).check()
            text_loader = TextLoader(file_path, encoding="utf-8")
            docx_loader = DocxLoader(file_path)
            doc_type = os.path.splitext(file_path)[-1]
            if doc_type == ".md":
                documents = text_loader.load()
            elif doc_type == ".docx":
                documents = docx_loader.load()
            elif doc_type == ".txt":
                documents = text_loader.load()
            else:
                documents = []
            docs = []
            for document in documents:
                docs.append(Document(page_content=document.page_content, metadata=document.metadata))
            return docs

        def execute_callback(split_texts: List[str]):
            if isinstance(filter_func, Callable):
                filter_texts = filter_func(split_texts)
                if not validata_list_str(filter_texts, [1, TEXT_MAX_LEN], [1, STR_MAX_LEN]):
                    logger.error(f"The return value of the callback method is not List[str], use raw doc slice")
                    return split_texts
                else:
                    return filter_texts
            return split_texts

        doc_cnt = 0
        split_doc_list = []
        file_types = ['*.txt', '*.docx', "*.md"]

        doc_set = set()
        for file_type in file_types:
            for doc_file in Path(document_path).glob(file_type):
                if doc_cnt > MAX_FILE_PROCESS_TIMES:
                    logger.warning(f"unable to process files over {MAX_FILE_PROCESS_TIMES} times")
                    break
                if not doc_file.is_file():
                    continue
                for doc in doc_load(doc_file.as_posix()):
                    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                              chunk_overlap=chunk_overlap)
                    texts = execute_callback(splitter.split_text(doc.page_content))
                    # 去重
                    unique_docs = [x for x in texts if x not in doc_set and not doc_set.add(x)]
                    split_doc_list.extend(unique_docs)
                doc_cnt = doc_cnt + 1

        return split_doc_list

    def generate_coarsest_qd_pairs(self,
                                   split_doc_list: list[str],
                                   question_number: int,
                                   prompt: str,
                                   chunk_size: int = 512):
        logger.info("query document pair generation")
        if len(split_doc_list) > MAX_DATASET_LEN:
            logger.error(f"inputs len should not bigger than {MAX_DATASET_LEN}, now is {len(split_doc_list)}")
            return [], []

        query_list = []
        doc_list = []
        origin_train_data_path = os.path.join(self.dataset_path, "origin_train_data.jsonl")
        if not os.path.exists(origin_train_data_path):
            query_list, doc_list = self._generate_qd_pairs(
                split_doc_list, question_number, origin_train_data_path, prompt, chunk_size
            )
        else:
            logger.info("The qd file is existed, check whether the next generation is required.")
            qd_pairs = read_jsonl_from_file(origin_train_data_path)

            for qd in qd_pairs:
                query_list.append(qd["anchor"])
                doc_list.append(qd["positive"])
            interrupted = doc_list[-1]
            interrupted_index = split_doc_list.index(interrupted)
            if interrupted_index == len(split_doc_list) - 1:
                logger.info("qd pairs generate finished, skip the generation process")
            else:
                logger.info("qd pairs generate not finished, continue to process")
                remain_doc_list = split_doc_list[(interrupted_index + 1):]
                new_query_list, new_doc_list = self._generate_qd_pairs(
                    remain_doc_list, question_number, origin_train_data_path, prompt, chunk_size
                )
                query_list.extend(new_query_list)
                doc_list.extend(new_doc_list)

        # 去重
        deduplicate_seen = set()
        deduplicate_queries = []
        deduplicate_docs = []
        for query, doc in zip(query_list, doc_list):
            if query not in deduplicate_seen:
                deduplicate_seen.add(query)
                deduplicate_queries.append(query)
                deduplicate_docs.append(doc)
        logger.info(f'remove duplicate queries len is {len(deduplicate_queries)}')
        return deduplicate_queries, deduplicate_docs

    def feature_qd_pair(self, query_list: list[str], doc_list: list[str],
                        reranker: LocalReranker, featured_percentage: float):
        """文档精选，使用bm25和reranker共同打分，按比率保留前面的问答对"""
        if not (1 > featured_percentage > 0):
            raise ValueError("featured_percentage must 0 ~ 1 range")
        logger.info("Selection-bm25 Scoring")
        bm25_scores_path = os.path.join(self.dataset_path, 'bm25_scores.jsonl')
        bm25_scores = []
        if os.path.exists(bm25_scores_path):
            datas = read_jsonl_from_file(bm25_scores_path)
            if len(datas) == len(query_list):
                bm25_scores = [data['score'] for data in datas]
        if len(bm25_scores) == 0:
            bm25_scores = bm25_featured(query_list, doc_list)
            # 保存bm25打分的分数
            datas = [{'anchor': query, 'positive': doc, 'score': score}
                     for query, doc, score in zip(query_list, doc_list, bm25_scores)]
            write_jsonl_to_file(datas, bm25_scores_path)
        bm25_sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_query_list = [query_list[i] for i in bm25_sorted_indices]

        logger.info("Selection-reranker Scoring")
        reranker_scores_path = os.path.join(self.dataset_path, 'reranker_scores.jsonl')
        reranker_scores = []
        if os.path.exists(reranker_scores_path):
            datas = read_jsonl_from_file(reranker_scores_path)
            if len(datas) == len(query_list):
                reranker_scores = [data['score'] for data in datas]
        if len(reranker_scores) == 0:
            reranker_scores = reranker_featured(reranker, query_list, doc_list)
            # 保存reranker打分的分数
            datas = [{'anchor': query, 'positive': doc, 'score': score}
                     for query, doc, score in zip(query_list, doc_list, reranker_scores)]
            write_jsonl_to_file(datas, reranker_scores_path)
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
        # 将排序后的元组列表解包并拆解成两个可迭代对象
        sorted_query_list, sorted_doc_list = zip(*sorted_zipped_lists)

        logger.info(f"Select the top {featured_percentage * 100}% data as the featured set based "
                    f"on the set parameters")
        featured_query_list = list(sorted_query_list[:round(len(sorted_query_list) * featured_percentage)])
        featured_doc_list = list(sorted_doc_list[:round(len(sorted_doc_list) * featured_percentage)])

        return featured_query_list, featured_doc_list

    def prefer_qd_pair(self, featured_query_list: list[str], featured_doc_list: list[str],
                       llm_threshold_score: float, prompt: str, chunk_size: int = 500):
        """大模型精选"""
        if not (1 > llm_threshold_score > 0):
            raise ValueError("featured_percentage must 0 ~ 1 range")
        logger.info("LLM score and eliminate those whose scores are lower than the preset threshold")
        llm_scores_path = os.path.join(self.dataset_path, 'llm_scores.jsonl')
        llm_scores = []
        if os.path.exists(llm_scores_path):
            scored_data_list = read_jsonl_from_file(llm_scores_path)
            scored_query_list = [data['anchor'] for data in scored_data_list]

            interrupted = scored_query_list[-1]
            interrupted_index = featured_query_list.index(interrupted)
            llm_scores = [data['score'] for data in scored_data_list]
            if interrupted_index == len(featured_query_list) - 1:
                logger.info("LLM scoring finished, skip the LLM scoring process")
            else:
                logger.info("LLM scoring not finished, continue to LLM scoring process")
                remain_query_list = featured_query_list[(interrupted_index + 1):]
                remain_doc_list = featured_doc_list[(interrupted_index + 1):]
                scores = self._prefer_scoring(remain_query_list, remain_doc_list, llm_scores_path, prompt, chunk_size)
                llm_scores.extend(scores)
        else:
            llm_scores = self._prefer_scoring(
                featured_query_list, featured_doc_list, llm_scores_path, prompt, chunk_size
            )
        # 使用列表推导式筛选出所有低于阈值分数的数据，并统计筛选结果的长度
        count_upper_threshold_score = len([x for x in llm_scores if x >= llm_threshold_score])

        llm_sorted_indices = sorted(range(len(llm_scores)), key=lambda i: llm_scores[i], reverse=True)
        llm_query_list = [featured_query_list[i] for i in llm_sorted_indices]
        llm_doc_list = [featured_doc_list[i] for i in llm_sorted_indices]

        preferred_query_list = llm_query_list[:count_upper_threshold_score]
        preferred_doc_list = llm_doc_list[:count_upper_threshold_score]

        return preferred_query_list, preferred_doc_list

    def _prefer_scoring(self, query_list, doc_list, llm_scores_path, prompt, chunk_size: int):
        logger.info(f"prefer scoring count: {len(query_list)}")
        score_list = []
        count = 0
        for i in range(0, len(query_list), chunk_size):
            chunk_query_list = query_list[i:i + chunk_size]
            chunk_doc_list = doc_list[i:i + chunk_size]
            llm_scores = llm_preferred(self.llm, chunk_query_list, chunk_doc_list, prompt)
            score_list.extend(llm_scores)
            qd_pair_scores = [{'anchor': query, 'positive': doc, 'score': score}
                              for query, doc, score in zip(chunk_query_list, chunk_doc_list, llm_scores)]
            write_jsonl_to_file(qd_pair_scores, llm_scores_path, 'a')
            logger.info(f"The {count + 1}st LLM scoring success")
            count += 1

        return score_list

    def _generate_qd_pairs(self,
                           split_doc_list: list[str],
                           question_number: int,
                           origin_train_data_path: str,
                           prompt: str,
                           chunk_size: int):
        logger.info(f"query document pair generation {len(split_doc_list)}")
        query_list = []
        doc_list = []
        count = 0
        for i in range(0, len(split_doc_list), chunk_size):
            chunk_doc_list = split_doc_list[i:i + chunk_size]  # 切片获取当前块的数据
            doc_queries = generate_qa_embedding_pairs(self.llm, chunk_doc_list, prompt, question_number)
            qd_pairs = []
            for doc, queries in doc_queries.items():
                query_list.extend(queries)
                docs = [doc] * len(queries)
                doc_list.extend(docs)
                for query, pos_doc in zip(queries, docs):
                    qd_pairs.append({"anchor": query, "positive": pos_doc})
            # 按块写文件
            write_jsonl_to_file(qd_pairs, origin_train_data_path, 'a')
            logger.info(f"The {count + 1}st query document pair generated success")
            count += 1

        return query_list, doc_list
