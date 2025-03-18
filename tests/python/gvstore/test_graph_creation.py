import os.path
import unittest
from unittest.mock import MagicMock, patch

from nebula3.gclient.net import Session

from mx_rag.gvstore.graph_creator.graph_create import GraphCreation
from mx_rag.gvstore.graph_creator.nebula_graph import NebulaGraph
from mx_rag.gvstore.graph_creator.vdb.vector_db import GraphVecMindfaissDB
from mx_rag.gvstore.util.utils import KgOprMode, safe_read_graphml, check_graph_name, check_entity_validity, \
    filter_graph_node, filter_graph_edge
from mx_rag.llm import Text2TextLLM

current_dir = os.path.dirname(os.path.realpath(__file__))
GRAPHML_FILE = os.path.realpath(os.path.join(current_dir, "../../../tests/data/test.graphml"))


class TestGvCreation(unittest.TestCase):
    def setUp(self):
        if os.path.exists(GRAPHML_FILE):
            os.remove(GRAPHML_FILE)

    @patch("nebula3.data.ResultSet")
    @patch("nebula3.gclient.net.Session", spec=Session)
    def test_graph_create(self, session, result_set):
        llm_mock = MagicMock(spec=Text2TextLLM)
        llm_mock.chat.return_value = ('{"Triplets": [["科学家", "发现", "新的行星"]], '
                                      '"Entity": {"科学家": "人", "新的行星": "科学成果"}, "Summary": "科学家发现行星。"}')
        graph_creation = GraphCreation(llm=MagicMock(spec=Text2TextLLM))
        chunks_dict = {}
        para_node = {
            "id": 0,
            "label": "text",
            "level": 0,
            "info": ["这个是一个测试的chunk，切分后的chunk"]
        }
        para_node1 = {
            "id": 1,
            "label": "text",
            "level": 0,
            "info": ["知识图谱要根据chunk来抽取三元组和实体"]
        }
        chunks_dict[0] = para_node
        chunks_dict[1] = para_node1
        file_contents = {"file_name1": chunks_dict}
        kwargs = {"lang": "zh", "graph_name": "graph_test", "vector_db": MagicMock(spec=GraphVecMindfaissDB)}
        graph_creation.graph_create(GRAPHML_FILE, file_contents, **kwargs)
        update_data = graph_creation.graph_update(GRAPHML_FILE, file_contents, KgOprMode.NEW, **kwargs)
        self.assertEqual(update_data.added_nodes[0], {'id': 0, 'label': 'file', 'info': 'file_name1'})
        self.assertEqual(update_data.added_edges[0], {'src': 0, 'dst': 1, 'name': '包含文本'})
        graph = safe_read_graphml(GRAPHML_FILE)
        self.assertEqual(len(graph.nodes.data()), 3)
        # 测试NebulaGraph
        result_set.row_size.return_value = 0
        result_set.is_succeeded.return_value = True
        session.execute.return_value = result_set
        vector_db = MagicMock(spec=GraphVecMindfaissDB)
        nebula = NebulaGraph(GRAPHML_FILE, GRAPHML_FILE, session=session, vector_db=vector_db)
        nebula.create_graph_index()
        nebula.update_graph_index(update_data)
        nebula.get_sub_graph([1], ["entity"], 1)
        self.assertEqual(nebula.get_nodes("hello"), ([], []))

    def test_check_graph_name(self):
        self.assertTrue(check_graph_name("graph_rag_123"))
        self.assertTrue(check_graph_name("GRAPH_WIKI1"))
        self.assertFalse(check_graph_name("graph_rag_123!@#"))
        self.assertFalse(check_graph_name("/tests/path/test.file"))
        self.assertFalse(check_graph_name(";select * from table;"))

    def test_check_entity_validity(self):
        self.assertTrue(check_entity_validity("小明"))
        self.assertTrue(check_entity_validity("info"))
        self.assertFalse(check_entity_validity(";select * from table;"))
        self.assertFalse(check_entity_validity("\'escape from str"))
        self.assertFalse(check_entity_validity("\"escape from str"))
        self.assertFalse(check_entity_validity("update tables"))

    def test_filter_graph_node(self):
        _, _, info = filter_graph_node(0, "text", "an American writer")
        self.assertEqual(info, "an American writer")
        self.assertEqual(filter_graph_node("1", "text", "an American writer"), ('1', 'text', 'an American writer'))
        self.assertEqual(filter_graph_node(1, "person", "an American writer"), (None, None, None))
        _, _, info = filter_graph_node(1, "entity", "an American writer\\")
        self.assertEqual(info, "an American writer")

    def test_filter_graph_edge(self):
        _, _, name = filter_graph_edge(0, 1, "belong to")
        self.assertEqual(name, "belong to")
        _, _, name = filter_graph_edge(0, 1, "belong to\\\\")
        self.assertEqual(name, "belong to")
        self.assertEqual(filter_graph_edge("0", 1, "belong to"), ('0', '1', 'belong to'))
        llm_mock = MagicMock(spec=Text2TextLLM)
        print(isinstance(llm_mock, Text2TextLLM))
