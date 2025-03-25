import unittest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from mx_rag.chain import Img2ImgChain
from mx_rag.llm import Img2ImgMultiModel, LLMParameterConfig
from mx_rag.retrievers import Retriever


class TestImgChain(unittest.TestCase):
    def test_query(self):
        model = MagicMock(spec=Img2ImgMultiModel)
        retriever = MagicMock(spec=Retriever)
        chain = Img2ImgChain(model, retriever)
        llm_config = MagicMock(spec=LLMParameterConfig)
        # 检索文档为空
        retriever.invoke.return_value = [Document("")]
        result = chain.query("question", llm_config)
        self.assertEqual(result, {})
        # 检索文档不为空
        retriever.invoke.return_value = [Document("这是被切分的chunk")]
        model.img2img.return_value = "模型返回结果"
        result = chain.query("question", llm_config)
        self.assertEqual(result, "模型返回结果")
