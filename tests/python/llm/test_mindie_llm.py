# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
import os
import unittest
from unittest import mock
from unittest.mock import patch
from loguru import logger

from mx_rag.llm import Text2TextLLM
from mx_rag.llm.llm_parameter import LLMParameterConfig


class MockResponse:
    def __init__(self, json_data, headers, status):
        self.json_data = json_data
        self.headers = headers
        self.status = status

    def stream(self, chunk_size):
        for content in self.json_data["choices"][0]["delta"]["content"]:
            mock_response = self.json_data.copy()
            mock_response["choices"][0]["delta"]["content"] = content
            yield bytes('data: ' + json.dumps(mock_response) + "\n", 'utf-8')
        mock_response = self.json_data.copy()
        mock_response["choices"][0]["delta"] = {}
        mock_response["choices"][0]["finish_reason"] = "stop"
        yield bytes('data: ' + json.dumps(mock_response), 'utf-8')

    def read(self, amt):
        return bytes(json.dumps(self.json_data), 'utf-8')


CONTENT = ('根据题目所给信息，我们可以知道这道题考查的是《赵氏孤儿》中的人物。'
           '因此，我们需要回忆起这部作品中的主要人物。'
           '\n\n经过查阅资料和记忆，我们得知，《赵氏孤儿》中的人物有程婴、公孙杵臼等。'
           '而选项A正是《赵氏孤儿》的别名之一，因此答案为A。')

RESPONSE = {
    'id': '99',
    'object': 'chat.completion',
    'created': 1716561049,
    'model': 'llama2-7b-hf',
    'choices': [{
        'index': 0,
        'message': {'role': 'assistant',
                    'content': CONTENT},
        'finish_reason': 'stop'}],
    'usage': {'prompt_tokens': 46, 'completion_tokens': 78, 'total_tokens': 124}
}

RESPONSE_STREAM = {
    'id': '99',
    'object': 'chat.completion',
    'created': 1716561049,
    'model': 'llama2-7b-hf',
    'choices': [{
        'index': 0,
        'delta': {'role': 'assistant',
                  'content': CONTENT},
        'finish_reason': None}],
    'usage': {'prompt_tokens': 46, 'completion_tokens': 78, 'total_tokens': 124}
}


class TestMindieLLM(unittest.TestCase):

    current_dir = os.path.dirname(os.path.realpath(__file__))

    def test_chat(self):
        with patch("urllib3.PoolManager.request", mock.Mock(
                return_value=MockResponse(RESPONSE, {
                    "Content-Type": "application/json",
                    "Content-Length": 200
                }, 200))):
            llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888")
            data = llm_model.chat(
                query="程婴、公孙杵臼是____中的人物。\nA. 《赵氏孤儿》\nB. 《杀狗记》\nC. 《墙头马上》\nD. 《岳阳楼》",
                sys_messages=[],
                llm_config=LLMParameterConfig(max_tokens=1024)
            )
            self.assertEqual(data, CONTENT)

    def test_chat_stream(self):
        with patch("urllib3.PoolManager.request", mock.Mock(return_value=MockResponse(RESPONSE_STREAM, {
            "Content-Type": "text/event-stream",
        }, 200))):
            llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888")
            stream_data = llm_model.chat_streamly(
                query="程婴、公孙杵臼是____中的人物。\nA. 《赵氏孤儿》\nB. 《杀狗记》\nC. 《墙头马上》\nD. 《岳阳楼》",
                sys_messages=[],
                llm_config=LLMParameterConfig(max_tokens=1024)
            )
            for i, data in enumerate(stream_data):
                self.assertEqual(data, CONTENT[:i + 1])

    def test_chat_interrupt(self):
        with patch("urllib3.PoolManager.request", mock.Mock(return_value=MockResponse(RESPONSE_STREAM, {
            "Content-Type": "application/json",
            "Content-Length": 200
        }, 404))):
            llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888")
            data = llm_model.chat(query="你好", sys_messages=[], llm_config=LLMParameterConfig(max_tokens=1024))
            self.assertEqual(data, "")

    def test_chat_stream_interrupt(self):
        with patch("urllib3.PoolManager.request", mock.Mock(return_value=MockResponse(RESPONSE_STREAM, {
            "Content-Type": "text/event-stream",
        }, 404))):
            llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888")
            stream_data = llm_model.chat_streamly(query="你好", sys_messages=[],
                                                  llm_config=LLMParameterConfig(max_tokens=1024))
            data = False
            for _ in stream_data:
                data = True
            self.assertFalse(data)

    def test_chat_param_max_tokens(self):
        error = False
        llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888")
        try:
            llm_model.chat(query="你好", llm_config=LLMParameterConfig(max_tokens=0))
        except ValueError:
            error = True
        self.assertTrue(error)

    def test_chat_param_history(self):
        error = False
        llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888")
        sys_messages = [{"role": "users", "content": "test"}] * 101
        try:
            llm_model.chat(query="你好", sys_messages=sys_messages)
        except ValueError:
            error = True
        self.assertTrue(error)

    def test_chat_param_history_2(self):
        error = False
        llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888")
        sys_messages = [{"role": "users", "content": "test", "111": "1"}]
        try:
            llm_model.chat(query="你好", sys_messages=sys_messages)
        except ValueError:
            error = True
        self.assertFalse(error)

    def test_chat_param_history_3(self):
        error = False
        llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888")
        sys_messages = [{"role": "user", "contentcontentcontentcontent": "test", "111": "1"}]
        try:
            llm_model.chat(query="你好", sys_messages=sys_messages)
        except ValueError:
            error = False
        self.assertFalse(error)

    def test_chat_param_presence_penalty(self):
        error = False
        llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888")
        try:
            llm_model.chat(query="你好", llm_config=LLMParameterConfig(presence_penalty=-5.0))
        except ValueError:
            error = True
        self.assertTrue(error)

    def test_chat_param_presence_penalty_2(self):
        error = False
        llm_model = Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888")
        try:
            llm_model.chat(query="你好", llm_config=LLMParameterConfig(presence_penalty=-5.0))
        except ValueError:
            error = True
        self.assertTrue(error)

    def test_chat_with_cert(self):
        cart_file = os.path.join(self.current_dir, "../../data/root_ca.crt")
        cart_file = os.path.realpath(cart_file)
        try:
            Text2TextLLM(model_name="llama2-7b-hf", base_url="https://test:8888", cert_file=cart_file)
        except Exception as e:
            logger.info(e)


if __name__ == '__main__':
    unittest.main()
