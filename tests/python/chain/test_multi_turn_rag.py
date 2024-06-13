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
import numpy as np
from mx_rag.embedding.local.text_embedding import TextEmbedding
from mx_rag.llm import Text2TextLLM
from mx_rag.retrievers import Retriever
from mx_rag.vectorstore.faiss_npu import MindFAISS
from mx_rag.storage import SQLiteDocstore, Document
from mx_rag.knowledge import Knowledge
from mx_rag.chain import MultiText2TextChain


class MyTestCase(unittest.TestCase):
    def test_multi_turn_rag(self):
        if not is_torch_npu_available():
            return

        emb = TextEmbedding("/workspace/bge-large-zh/")
        db = SQLiteDocstore("/tmp/sql.db")
        logger.info("create emb done")
        MindFAISS.set_device(2)
        logger.info("set_device done")
        index = MindFAISS(x_dim=1024, index_type="FLAT:L2")
        vector_store = Knowledge("./sql.db", db, index, "test", white_paths=["/home"])
        vector_store._add_texts("test_file.txt", ["this is a test"], embed_func=emb.embed_texts)
        logger.info("create MindFAISS done")
        llm = Text2TextLLM(model_name="chatglm2-6b-quant", url="http://71.14.88.12:7890")
        r = Retriever(vector_store=vector_store, document_store= db, embed_func=emb.embed_texts)
        rag = MultiText2TextChain(retriever=r, llm=llm)
        response = rag.query("Please remember that Xiao Ming's father is Xiao Gang.", max_tokens=1024, temperature=1.0,
                             top_p=0.1)
        logger.debug(f"stream response {response}")

        for i in range(1, 22):
            response = rag.query(f"Who is Xiaoming's father? iter {i}", max_tokens=1024, temperature=1.0, top_p=0.1)
            logger.debug(f"stream response {response}")
        self.assertTrue(True)

    def test_multi_turn_rag_no_npu(self):
        if is_torch_npu_available():
            return

        def embed_func(texts):
            return np.random.random((1, 1024))


        db = SQLiteDocstore("sql.db")
        MindFAISS.DEVICES = MagicMock()
        index = MindFAISS(x_dim=1024, index_type="FLAT:L2")
        vector_store = Knowledge("./sql.db", db, index, "test", white_paths=["/home"])
        vector_store.similarity_search = MagicMock(
            return_value=[[(Document(page_content="this is a test", document_name="test.txt"), 0.5)]])
        llm = Text2TextLLM(model_name="chatglm2-6b-quant", url="http://127.0.0.1:7890")

        r = Retriever(vector_store=vector_store, document_store= db, embed_func=embed_func)
        r.get_relevant_documents = MagicMock(
            return_value=[Doc(page_content="this is a test", metadata={})])

        rag = MultiText2TextChain(retriever=r, llm=llm)
        llm.chat = MagicMock(return_value=("MultiQueryRetriever"))
        response = rag.query("Please remember that Xiao Ming's father is Xiao Gang.", max_tokens=1024, temperature=1.0,
                             top_p=0.1)
        logger.debug(f"stream response {response}")
        self.assertEqual("MultiQueryRetriever", response)

        response = rag.query("Who is Xiaoming's father?", max_tokens=1024, temperature=1.0, top_p=0.1)
        logger.debug(f"stream response {response}")
        self.assertEqual("MultiQueryRetriever", response)


if __name__ == '__main__':
    unittest.main()
