import unittest
from unittest.mock import MagicMock, patch

from paddle.base import libpaddle
from langchain_core.documents import Document
from mx_rag.chain import Multi2MultiChain
from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.retrievers import Retriever


class TestMulti2MultiChain(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.mock_llm = MagicMock(spec=Text2TextLLM)
        self.mock_retriever = MagicMock(spec=Retriever)
        self.chain = Multi2MultiChain(self.mock_llm, self.mock_retriever)
        self.mock_llm_config = MagicMock(spec=LLMParameterConfig)

    @patch('mx_rag.llm.Text2TextLLM')
    def test_query_with_no_documents(self, mock_llm):
        """测试没有检索到文档的情况"""
        self.mock_retriever.invoke.return_value = []
        self.mock_llm.chat.return_value = "模型返回结果"

        result = self.chain.query("question", self.mock_llm_config)

        expected_result = {
            'query': 'question',
            'result': '模型返回结果',
            'source_documents': []
        }
        self.assertEqual(result, expected_result, msg="当没有检索到文档时，结果应包含空的source_documents")

    @patch('mx_rag.llm.Text2TextLLM')
    def test_query_with_text_documents(self, mock_llm):
        """测试只检索到文本文档的情况"""
        self.mock_retriever.invoke.return_value = [
            Document(metadata={'source': 'path', 'type': 'text'}, page_content='这是被切分的chunk')
        ]
        self.mock_llm.chat.return_value = "模型返回结果"

        result = self.chain.query("question", self.mock_llm_config)

        expected_result = {
            'query': 'question',
            'result': '模型返回结果',
            'source_documents': [
                {'metadata': {'source': 'path', 'type': 'text'}, 'page_content': '这是被切分的chunk'}
            ]
        }
        self.assertEqual(result, expected_result, msg="当只检索到文本文档时，结果应包含相应的文本文档信息")

    @patch('mx_rag.llm.Text2TextLLM')
    def test_query_with_image_documents(self, mock_llm):
        """测试只检索到图片文档的情况"""
        self.mock_retriever.invoke.return_value = [
            Document(metadata={'source': 'path', 'type': 'image'}, page_content='这是被切分的chunk')
        ]
        self.mock_llm.chat.return_value = "模型返回结果"

        result = self.chain.query("question", self.mock_llm_config)

        expected_result = {
            'query': 'question',
            'result': '模型返回结果',
            'source_documents': [
                {'metadata': {'source': 'path', 'type': 'image'}, 'page_content': '这是被切分的chunk'}
            ]
        }
        self.assertEqual(result, expected_result, msg="当只检索到图片文档时，结果应包含相应的图片文档信息")


if __name__ == "__main__":
    unittest.main()
