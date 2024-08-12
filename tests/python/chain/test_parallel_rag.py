# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import time
import unittest
from unittest import mock
from unittest.mock import patch

from mx_rag.chain import ParallelText2TextChain


class TestParallelChain(unittest.TestCase):
    def test_init(self):
        parallel_chain = ParallelText2TextChain(None, None)
        self.assertIsInstance(parallel_chain, ParallelText2TextChain)

    def test_query_prefill_first_done(self):
        def mock_do_stream_query(*args, **kwargs):
            yield "prefill query done"

        def mock_retrieve_process(*args, **kwargs):
            time.sleep(0.2)

        with patch('mx_rag.chain.SingleText2TextChain._do_stream_query',
                   mock.Mock(side_effect=mock_do_stream_query)):
            with patch('mx_rag.chain.ParallelText2TextChain._retrieve_process',
                       mock.Mock(side_effect=mock_retrieve_process)):
                parallel_chain = ParallelText2TextChain(None, None)
                answer = parallel_chain.query("123456")
                self.assertEqual(answer, "prefill query done")
                self.assertEqual(parallel_chain.prefill_done.value, 0)

    def test_query_retrieve_first_done(self):
        def mock_do_stream_query(*args, **kwargs):
            time.sleep(0.2)
            yield "prefill query done"

        def mock_retrieve_process(*args, **kwargs):
            pass

        def mock_do_query(*args, **kwargs):
            return "retrieve query done"

        with patch('mx_rag.chain.SingleText2TextChain._do_stream_query',
                   mock.Mock(side_effect=mock_do_stream_query)):
            with patch('mx_rag.chain.ParallelText2TextChain._retrieve_process',
                       mock.Mock(side_effect=mock_retrieve_process)):
                with patch('mx_rag.chain.SingleText2TextChain._do_query',
                           mock.Mock(side_effect=mock_do_query)):
                    parallel_chain = ParallelText2TextChain(None, None)
                    answer = parallel_chain.query("123456")
                    self.assertEqual(answer, "retrieve query done")
                    self.assertEqual(parallel_chain.prefill_done.value, 0)


if __name__ == '__main__':
    unittest.main()
