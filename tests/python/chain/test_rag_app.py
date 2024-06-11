import os
import sys
import unittest
from unittest.mock import MagicMock

from transformers import is_torch_npu_available

from mx_rag.document.loader import DocxLoader

if not is_torch_npu_available():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(cur_dir, "../vectorstore/"))

from loguru import logger
from transformers import is_torch_npu_available

from mx_rag.embedding.local.text_embedding import TextEmbedding
from mx_rag.llm import MindieLLM
from mx_rag.retrievers import Retriever, MultiQueryRetriever
from mx_rag.vectorstore.faiss_npu import MindFAISS
from mx_rag.storage import SQLiteDocstore, Document
from mx_rag.chain import SimpleRetrieval
from mx_rag.document.splitter.char_text_splitter import CharTextSplitter


class MyTestCase(unittest.TestCase):
    sql_db_file = "/tmp/sql.db"
    def setUp(self):
        if os.path.exists(MyTestCase.sql_db_file):
            os.remove(MyTestCase.sql_db_file)

    def test_with_npu(self):
        if not is_torch_npu_available():
            return
        current_dir = os.path.dirname(os.path.realpath(__file__))

        loader = DocxLoader(os.path.realpath(os.path.join(current_dir, "../../data/mxVision.docx")))
        spliter = CharTextSplitter()
        res = loader.load_and_split(spliter)
        emb = TextEmbedding("/workspace/bge-large-zh/")
        db = SQLiteDocstore("/tmp/sql.db")
        logger.info("create emb done")
        MindFAISS.set_device(2)
        logger.info("set_device done")

        vector_store = MindFAISS(x_dim=1024, index_type="FLAT:L2", document_store=db, embed_func=emb.encode)

        vector_store.add_texts("mxVision.docx",
                               [d.page_content for d in MyTestCase.res],
                               [d.metadata for d in MyTestCase.res]
                               )

        logger.info("create MindFAISS done")
        llm = MindieLLM(model_name="chatglm2-6b-quant", url="http://71.14.88.12:7890")

        def test_rag_chain_npu_signle(self):
            r = Retriever(vector_store=vector_store, k=1, score_threshold=0.5)
            rag = SimpleRetrieval(retriever=r, llm=llm)
            good_prompt = "mxVision软件架构包含哪些模块？"
            final_ans = ""
            for response in rag.query(good_prompt, max_tokens=1024, temperature=0.1, top_p=1.0, stream=True):
                final_ans = response
            logger.debug(f"final_ans {final_ans}")

        def test_rag_chain_npu_multi_doc(self):
            multi_sr_prompt = "mxVision软件包介绍"
            r = Retriever(vector_store=vector_store, k=5, score_threshold=0.7)
            rag = SimpleRetrieval(retriever=r, llm=llm)
            final_ans = ""
            for response in rag.query(multi_sr_prompt, max_tokens=1024, temperature=0.1, top_p=1.0, stream=True):
                final_ans = response
            logger.debug(f"final_ans {final_ans}")

        def test_rag_chain_npu_no_doc(self):
            r = Retriever(vector_store=vector_store, score_threshold=0.5)
            rag = SimpleRetrieval(retriever=r, llm=llm)
            final_ans = ""
            for response in rag.query("CANN是什么呢", max_tokens=1024, temperature=0.1, top_p=1.0, stream=True):
                final_ans = response
            logger.debug(f"final_ans {final_ans}")

        test_rag_chain_npu_no_doc(self)


if __name__ == '__main__':
    unittest.main()
