# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import random

import faiss
import numpy as np
from loguru import logger
from tqdm import tqdm

from mx_rag.embedding.local import TextEmbedding
from mx_rag.utils.common import validate_params, MAX_DEVICE_ID, STR_TYPE_CHECK_TIP

QUERY = "query"
POS = "pos"
NEG = "neg"


class MineHardNegative:
    @validate_params(
        model=dict(validator=lambda x: isinstance(x, str), message=STR_TYPE_CHECK_TIP),
        dev_id=dict(validator=lambda x: isinstance(x, int) and 0 <= x <= MAX_DEVICE_ID,
                    message="param must be int and value range [0, 63]")
    )
    def __init__(self, model: str, dev_id: int = 0):
        self.model = TextEmbedding(model, dev_id=dev_id)

    @staticmethod
    def _create_index(embeddings: np.ndarray):
        index = faiss.IndexFlatIP(len(embeddings[0]))
        index.add(embeddings)
        return index

    @staticmethod
    def _batch_search(index,
                      query: np.ndarray,
                      top_k: int = 200,
                      batch_size: int = 64):
        all_scores, all_inxs = [], []
        for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
            batch_query = query[start_index:start_index + batch_size]
            batch_scores, batch_inxs = index.search(batch_query, k=top_k)
            all_scores.extend(batch_scores.tolist())
            all_inxs.extend(batch_inxs.tolist())
        return all_scores, all_inxs

    @validate_params(
        sample_range=dict(validator=lambda x: len(x) == 2, message="param length must be 2"),
        negative_number=dict(validator=lambda x: 1 <= x <= 10, message="param value range [1, 10]")
    )
    def find_knn_neg(self,
                     train_data: list[dict],
                     sample_range: list[int],
                     negative_number: int):
        corpus = []
        queries = []

        def check_list_str(data_list: list):
            for data in data_list:
                if not isinstance(data, str):
                    return False
            return True

        for data in train_data:
            data_query = data.get(QUERY)
            if isinstance(data_query, str):
                queries.append(data_query)

            data_pos = data.get(POS)
            if isinstance(data_pos, list) and check_list_str(data_pos):
                corpus.extend(data_pos)

            data_neg = data.get(NEG)
            if isinstance(data_neg, list) and check_list_str(data_neg):
                corpus.extend(data_neg)

        corpus = list(set(corpus))

        logger.info(f"inference embedding for corpus (number={len(corpus)})")
        p_vecs = np.array(self.model.embed_documents(corpus))

        logger.info(f"inference embedding for queries (number={len(queries)})")
        q_vecs = np.array(self.model.embed_documents(queries))

        logger.info("create index and search")
        index = self._create_index(p_vecs)
        _, all_inxs = self._batch_search(index, q_vecs, top_k=sample_range[-1])

        for i, data in enumerate(train_data):
            query = data[QUERY]
            inxs = all_inxs[i][sample_range[0]:sample_range[1]]
            filtered_inx = []
            for inx in inxs:
                if inx == -1:
                    break
                if corpus[inx] not in data[POS] and corpus[inx] != query:
                    filtered_inx.append(inx)

            if len(filtered_inx) > negative_number:
                filtered_inx = random.sample(filtered_inx, negative_number)
            data[NEG] = [corpus[inx] for inx in filtered_inx]

        for data in train_data:
            if len(data[NEG]) < negative_number and negative_number - len(data[NEG]) <= len(corpus):
                data[NEG].extend(random.sample(corpus, negative_number - len(data[NEG])))

        return train_data
