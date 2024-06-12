import os
import sys
import unittest
from unittest.mock import MagicMock

from transformers import is_torch_npu_available

from mx_rag.document.doc import Doc

if not is_torch_npu_available():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(cur_dir, "../vectorstore/"))

from loguru import logger
from transformers import is_torch_npu_available

from mx_rag.embedding.local.text_embedding import TextEmbedding
from mx_rag.llm import Text2TextLLM
from mx_rag.retrievers import Retriever, MultiQueryRetriever
from mx_rag.vectorstore.faiss_npu import MindFAISS
from mx_rag.storage import SQLiteDocstore, Document
import numpy as np
from mx_rag.chain import SingleText2TextChain


class MyTestCase(unittest.TestCase):
    sql_db_file = "/tmp/sql.db"

    def setUp(self):
        if os.path.exists(MyTestCase.sql_db_file):
            os.remove(MyTestCase.sql_db_file)

    def test_with_npu(self):
        if not is_torch_npu_available():
            return

        emb = TextEmbedding("/workspace/bge-large-zh/")
        db = SQLiteDocstore("/tmp/sql.db")
        logger.info("create emb done")
        MindFAISS.set_device(2)
        logger.info("set_device done")
        vector_store = MindFAISS(x_dim=1024, index_type="FLAT:L2", document_store=db)
        vector_store.add_texts("test_file.txt", ["this is a test"], emb.embed_texts, [{"filepath": "xxx.file"}])
        logger.info("create MindFAISS done")
        llm = Text2TextLLM(model_name="chatglm2-6b-quant", url="http://71.14.88.12:7890")

        def test_rag_chain_npu(self):
            r = Retriever(vector_store=vector_store, embed_func=emb.embed_texts)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            response = rag.query("who are you??", max_tokens=1024, temperature=1.0, top_p=0.1)
            logger.debug(f"response {response}")

        def test_rag_chain_npu_stream(self):
            r = Retriever(vector_store=vector_store, embed_func=emb.embed_texts)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            for response in rag.query("who are you??", max_tokens=1024, temperature=1.0, top_p=0.1,
                                      stream=True):
                logger.debug(f"stream response {response}")

        def test_rag_chain_npu_multi_query_retriever(self):
            r = MultiQueryRetriever(llm, vector_store=vector_store, embed_func=emb.embed_texts)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            response = rag.query("who are you??", max_tokens=1024, temperature=1.0, top_p=0.1)
            logger.debug(f"response {response}")

        def test_rag_chain_npu_stream_multi_query_retriever(self):
            r = MultiQueryRetriever(llm, vector_store=vector_store, embed_func=emb.embed_texts)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            rag.source = True
            for response in rag.query("who are you??", max_tokens=1024, temperature=1.0, top_p=0.1,
                                      stream=True):
                logger.debug(f"stream response {response}")

        test_rag_chain_npu(self)
        test_rag_chain_npu_stream(self)
        test_rag_chain_npu_multi_query_retriever(self)
        test_rag_chain_npu_stream_multi_query_retriever(self)

    def test_with_no_npu(self):
        if is_torch_npu_available():
            return

        def embed_func(texts):
            return np.random.random((1, 1024))

        db = SQLiteDocstore("sql.db")
        MindFAISS.DEVICES = MagicMock()
        vector_store = MindFAISS(x_dim=1024, index_type="FLAT:L2", document_store=db)
        vector_store.similarity_search = MagicMock(
            return_value=[(Document(page_content="this is a test", document_name="test.txt"), 0.5)])
        llm = Text2TextLLM(model_name="chatglm2-6b-quant", url="http://127.0.0.1:7890")

        def test_rag_chain_npu(self):
            r = Retriever(vector_store=vector_store, embed_func=embed_func)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            llm.chat = MagicMock(return_value="test test test")
            response = rag.query("who are you??", max_tokens=1024, temperature=1.0, top_p=0.1)
            self.assertEqual("test test test", response)

        def test_rag_chain_npu_stream(self):
            r = Retriever(vector_store=vector_store, embed_func=embed_func)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            llm.chat_streamly = MagicMock(return_value=(yield "Retriever steam"))
            for response in rag.query("who are you??", max_tokens=1024, temperature=1.0, top_p=0.1,
                                      stream=True):
                self.assertEqual("Retriever steam", response)

        def test_rag_chain_npu_multi_query_retriever(self):
            r = MultiQueryRetriever(llm, vector_store=vector_store, embed_func=embed_func)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            llm.chat = MagicMock(return_value=("MultiQueryRetriever"))
            response = rag.query("who are you??", max_tokens=1024, temperature=1.0, top_p=0.1)
            self.assertEqual("MultiQueryRetriever", response)

        def test_rag_chain_npu_stream_multi_query_retriever(self):
            r = MultiQueryRetriever(llm, vector_store=vector_store, embed_func=embed_func)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            rag.source = True
            llm.chat_streamly = MagicMock(return_value=(yield "MultiQueryRetriever steam"))
            for response in rag.query("who are you??", max_tokens=1024, temperature=1.0, top_p=0.1,
                                      stream=True):
                self.assertEqual("MultiQueryRetriever steam", response)

        def test_merge_query_prompt(self):
            r = Retriever(vector_store=vector_store, embed_func=embed_func)
            rag = SingleText2TextChain(retriever=r, llm=llm)
            query = "who are you?"
            prompt = "请根据上面提示输出对应结果"
            res = rag._merge_query_prompt(query, docs=[], prompt=prompt)
            self.assertEqual(res, query)

            docs = [Doc(page_content = "you are a smart assistants", metadata= {})]
            res = rag._merge_query_prompt(query, docs=docs, prompt=prompt)
            self.assertEqual(res, "\n\n".join(("you are a smart assistants", prompt, query)))

        test_rag_chain_npu(self)
        test_rag_chain_npu_stream(self)
        test_rag_chain_npu_multi_query_retriever(self)
        test_rag_chain_npu_stream_multi_query_retriever(self)
        test_merge_query_prompt(self)



if __name__ == '__main__':
    unittest.main()
