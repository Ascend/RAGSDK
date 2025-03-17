import os
import unittest
from unittest.mock import MagicMock, patch

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nebula3.gclient.net import Session

from mx_rag.document import LoaderMng
from mx_rag.document.loader import DocxLoader
from mx_rag.embedding.local import TextEmbedding
from mx_rag.gvstore.graph_creator import KGEngine
from mx_rag.gvstore.retrieval.retriever.graph_retrieval import GraphRetriever
from mx_rag.gvstore.util.utils import KgOprMode
from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.reranker.local import LocalReranker
from mx_rag.storage.vectorstore import MindFAISS

current_dir = os.path.dirname(os.path.realpath(__file__))
WORK_DIR = os.path.realpath(os.path.join(current_dir, "../../../tests/data"))
GRAPHML_FILE = os.path.realpath(os.path.join(current_dir, "../../../tests/data/test.graphml"))
GRAPH_NAME = "graph_wiki"


class TestGvRetriever(unittest.TestCase):
    def setUp(self):
        graphml_file = os.path.join(WORK_DIR, f"{GRAPHML_FILE}.graphml")
        sql_file = os.path.join(WORK_DIR, "sql.db")
        if os.path.exists(graphml_file):
            os.remove(graphml_file)
        if os.path.exists(sql_file):
            os.remove(sql_file)
        llm_mock = MagicMock(spec=Text2TextLLM)
        llm_mock.chat.return_value = ('{"Triplets": [["科学家", "发现", "新的行星"]], '
                                      '"Entity": {"科学家": "人", "新的行星": "科学成果"}, "Summary": "科学家发现行星。"}')
        llm_mock.llm_config = MagicMock(spec=LLMParameterConfig)
        llm_mock = llm_mock
        embed = MagicMock(spec=TextEmbedding)
        reranker = MagicMock(spec=LocalReranker)
        mind_faiss = MagicMock(spec=MindFAISS)
        mind_faiss.search.return_value = ([[0.9999486207962036]], [[0]])
        session = MagicMock(spec=Session)
        self.kg_creation = KGEngine(llm=llm_mock, embedding_model=embed, rerank_model=reranker,
                                    vector_db=mind_faiss, work_dir=WORK_DIR, lang="en",
                                    nebula_session=session)

    @patch("mx_rag.gvstore.retrieval.retriever.graph_retrieval.GraphRetriever", spec=GraphRetriever)
    def test_upload_kg_files(self, graph_retriever):
        loader_mng = LoaderMng()
        loader_mng.register_loader(loader_class=DocxLoader, file_types=[".docx"])
        # 加载文档加载器，可以使用mxrag自有的，也可以使用langchain的
        loader_mng.register_loader(loader_class=TextLoader, file_types=[".txt"])
        # 加载文档切分器，使用langchain的
        loader_mng.register_splitter(splitter_class=RecursiveCharacterTextSplitter,
                                     file_types=[".docx", ".txt"],
                                     splitter_params={"chunk_size": 512,
                                                      "chunk_overlap": 20,
                                                      "separators": [".", "。", "!", "！", "?", "？", "\n"]})
        self.kg_creation.upload_kg_files([os.path.join(WORK_DIR, "test.txt")], loader_mng)
        self.kg_creation.create_kg_graph(GRAPH_NAME, entity_filter_flag=True)
        self.kg_creation.upload_kg_files([os.path.join(WORK_DIR, "link.docx")], loader_mng)
        self.kg_creation.update_kg_graph(GRAPH_NAME, KgOprMode.NEW, entity_filter_flag=True)
        retriever = self.kg_creation.as_retriever(GRAPH_NAME)
        self.assertTrue(isinstance(retriever, GraphRetriever))
        # 无法mock覆盖到GraphRetriever里的方法，和pydantic有关
        graph_retriever.retrieval.return_value = []
        self.assertEqual(self.kg_creation.retrival_kg_graph(GRAPH_NAME, "测试问题？"), ["test.txt"])

