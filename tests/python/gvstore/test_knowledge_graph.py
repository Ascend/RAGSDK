import os
import shutil
import unittest
from paddle.base import libpaddle
from langchain_community.document_loaders import TextLoader
from unittest.mock import patch
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mx_rag.document import LoaderMng
from mx_rag.document.loader import DocxLoader, PdfLoader
from mx_rag.embedding.local import TextEmbedding
from mx_rag.gvstore.graph_creator.kg_engine import KGEngine
from mx_rag.gvstore.util.utils import KgOprMode, safe_read_graphml
from mx_rag.libs.glib.utils.file_utils import FileCreate, FileCheck
from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.reranker.local import LocalReranker
from mx_rag.storage.vectorstore import MilvusDB
from mx_rag.utils import ClientParam


class TestKGCreateCase(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.realpath(os.path.join(current_dir, "../../data"))
        self.data_dir = data_dir
        self.file_list = [os.path.join(data_dir, "test.txt"),
                     os.path.join(data_dir, "demo.docx"),
                     os.path.join(data_dir, "test_cn.pdf")]
        self.work_dir = os.path.join(current_dir, "tmp")

    @patch("mx_rag.llm.Text2TextLLM.chat")
    @patch("mx_rag.storage.vectorstore.MilvusDB", spec=MilvusDB)
    @patch("mx_rag.reranker.local.LocalReranker", spec=LocalReranker)
    @patch("mx_rag.embedding.local.TextEmbedding", spec=TextEmbedding)
    def test_gvstore(self, emb, reranker, milvus_db, llm_chat):
        FileCreate.create_dir(self.work_dir, 0o777)
        graph_name = "test_txt"
        llm_config = LLMParameterConfig(temperature=0.1, max_tokens=2000)
        llm = Text2TextLLM(model_name="llama2-7b-hf", base_url="http://test:8888",
                           llm_config=llm_config, client_param=ClientParam(use_http=True))
        kg_creation = KGEngine(llm, emb, reranker, milvus_db, self.work_dir)
        loader_mng = LoaderMng()
        # 加载文档加载器，可以使用mxrag自有的，也可以使用langchain的
        loader_mng.register_loader(loader_class=TextLoader, file_types=[".txt"])
        loader_mng.register_loader(loader_class=DocxLoader, file_types=[".docx"])
        loader_mng.register_loader(loader_class=PdfLoader, file_types=[".pdf"])
        # 加载文档切分器，使用langchain的
        loader_mng.register_splitter(splitter_class=RecursiveCharacterTextSplitter,
                                     file_types=[".txt", ".docx", ".pdf"],
                                     splitter_params={"chunk_size": 512,
                                                      "chunk_overlap": 20,
                                                      "separators": [".", "。", "!", "！", "?", "？", "\n"]})

        def side_effect(documents):
            # 根据 documents 的内容返回不同的值
            return [[1, 2]] * len(documents)

        def test_kg_create(self):
            result = kg_creation.upload_kg_files(self.file_list, loader_mng)
            self.assertTrue(result)
            x_dim = 1024
            llm_chat.return_value = "{'Triplets': [['小明', '和', '小红'], ['小明', '去看', '电影'], " \
                                    "['小明', '找', '小花'], ['小花', '去', '图书馆']], " \
                                    "'Entity': {'小明': '人', '小红': '人', '电影': '娱乐', '饭': '食物', '小花': '人', '图书馆': '场所'}, " \
                                    "'Summary': '小明与朋友看电影、吃饭、去图书馆。'}"
            emb.embed_documents.side_effect = side_effect
            kg_creation.create_kg_graph(graph_name, x_dim=x_dim)
            graph_path = os.path.join(self.work_dir, f"{graph_name}.graphml")
            FileCheck.check_input_path_valid(graph_path)
            graph = safe_read_graphml(graph_path)

            self.assertEqual(len(graph.nodes.data()), 106)
            self.assertEqual(len(graph.edges.data()), 453)

        def test_kg_update(self):
            update_files = [os.path.join(self.data_dir, "update_text.txt")]
            result = kg_creation.upload_kg_files(update_files, loader_mng)
            self.assertTrue(result)
            llm_chat.return_value = """
            {"Triplets": [["小明", "因为", "多科考试不及格"],["小明", "被", "老师训了一顿"],["老师", "要求", "小明明天请家长来学校一趟"]],  
            "Entity": {"小明": "学生", "老师": "教师", "多科考试不及格": "考试结果","家长": "家庭成员" },
            "Summary": "小明考试不及格被老师训，要求请家长来学校。"}
            """
            kg_creation.update_kg_graph(graph_name, KgOprMode.NEW)
            graph_path = os.path.join(self.work_dir, f"{graph_name}.graphml")
            FileCheck.check_input_path_valid(graph_path)
            graph = safe_read_graphml(graph_path)
            self.assertEqual(len(graph.nodes.data()), 112)
            self.assertEqual(len(graph.edges.data()), 462)

        @patch("mx_rag.reranker.local.LocalReranker.rerank_top_k")
        @patch("mx_rag.reranker.local.LocalReranker.rerank")
        @patch("mx_rag.gvstore.graph_creator.vdb.vector_db.MilvusVecDB.get_data_by_ids")
        @patch("mx_rag.gvstore.graph_creator.vdb.vector_db.MilvusVecDB.search_with_docs")
        def test_kg_query(self, vdb_search, get_data_by_ids, rerank, rerank_top_k):
            question = "小明和小花干什么"
            retrieval_k = 3
            k_hop = 2
            vdb_search.return_value = [0.2, 0.1, 0.7], [1, 2, 3], ["小明和小红去看电影", "小明去看电影，看完电影吃饭",
                                                                   "小明去找小花一块去图书馆"]
            get_data_by_ids.return_value = [1, 2], ["小明和小红去看电影", "小明去找小花一块去图书馆"]
            rerank.return_value = [0.1, 0.2, 0.3]
            rerank_top_k.rerank_top_k = ["小明和小红去看电影", "小明去找小花一块去图书馆"]
            context = kg_creation.retrieval_kg_graph(graph_name, question, top_k=retrieval_k, k_hop=k_hop)
            self.assertTrue("小明去找小花一块去图书馆" in context)

        test_kg_create(self)
        test_kg_update(self)
        test_kg_query(self)

    def tearDown(self):
        shutil.rmtree(self.work_dir)


if __name__ == '__main__':
    unittest.main()
