import jieba
from rank_bm25 import BM25Okapi
from typing import List, Dict
from langchain_core.embeddings import Embeddings


class BM25Embedding(Embeddings):

    def _fit_bm25(self, documents: List[str]):
        """
        训练 BM25 模型，给定文档集合。

        :param documents: 文档列表
        """
        self.documents = documents
        self.tokenized_documents = [self._preprocess(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_documents)

    def _preprocess(self, text: str) -> List[str]:
        """
        使用结巴对文本进行分词

        :param text: 需要分词的文本
        :return: 分词后的列表
        """
        return list(jieba.cut(text))

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[Dict[int, float]]:
        """
        对文档集合进行稀疏向量化

        :param texts: 文本集合
        :param batch_size: 批次大小
        :return: 每个文档的稀疏向量（字典格式）
        """
        result = []
        for start_index in range(0, len(texts), batch_size):
            batch_texts = texts[start_index:start_index + batch_size]
            batch_result = self._encode(batch_texts)
            result.extend(batch_result)
        return result

    def embed_query(self, text: str) -> Dict[int, float]:
        """
        对单个查询文本进行稀疏向量化

        :param text: 查询文本
        :return: 查询文本的稀疏向量
        """
        result = self._encode([text])
        print('embed_query_result', result)
        return result[0]

    def _encode(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        对每个文本进行稀疏向量化，返回稀疏向量（只保留非零分数）。

        :param texts: 文本集合
        :return: 每个文本的稀疏向量
        """
        self._fit_bm25(texts)  # 初始化 BM25 模型
        result = []
        for text in texts:
            # 获取文本的BM25分数
            doc_tokens = self._preprocess(text)
            scores = self.bm25.get_scores(doc_tokens)
            print('doc_tokens', doc_tokens)
            # 生成稀疏向量，只保留非零分数
            doc_sparse_vector = {idx: score for idx, score in enumerate(scores) if score > 0}
            result.append(doc_sparse_vector)
        return result
