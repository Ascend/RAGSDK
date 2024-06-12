import unittest

from mx_rag.chain import Chain
from mx_rag.chain.router import QueryRouter, TextClassifier


class MockClassifier(TextClassifier):
    def classify(self, text, labels):
        """ classify labels """
        return "text generate text"

class MockImg2ImgChain(Chain):
    def query(self, text, *args, **kwargs):
        return "this is test case"

class TestQueryRouter(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_chain_register_and_unregister(self):
        q = QueryRouter(MockClassifier())
        chain = MockImg2ImgChain()
        # 注册不允许的label
        q.register_chain("audio generate text", chain)
        labels = q.get_register_labels()
        self.assertSequenceEqual(labels, [])

        q.register_chain("text generate text", chain)
        labels = q.get_register_labels()
        self.assertSequenceEqual(labels, ["text generate text"])
        # 注册重复的label
        q.register_chain("text generate text", chain)
        labels = q.get_register_labels()
        self.assertSequenceEqual(labels, ["text generate text"])

        q.register_chain("text generate image", chain)
        labels = q.get_register_labels()
        self.assertSequenceEqual(labels, ["text generate text", "text generate image"])

        q.register_chain("image generate image", chain)
        labels = q.get_register_labels()
        self.assertSequenceEqual(labels, ["text generate text", "text generate image", "image generate image"])
        # 删除不存在的label
        q.unregister_chain("audio generate text")
        labels = q.get_register_labels()
        self.assertSequenceEqual(labels, ["text generate text", "text generate image", "image generate image"])
        labels = q.get_register_labels()
        # 删除存在的label
        q.unregister_chain("text generate image")
        self.assertSequenceEqual(labels, ["text generate text", "image generate image"])

    def test_route_to_llm(self):
        q = QueryRouter(MockClassifier())
        chain = MockImg2ImgChain()
        q.register_chain("text generate image", chain)

        res = q.route_to_llm("安装cann依赖软件有哪些？")
        self.assertEqual(res, "")


        q.register_chain("text generate text", chain)

        res = q.route_to_llm("安装cann依赖软件有哪些？")
        self.assertEqual(res, "this is test case")