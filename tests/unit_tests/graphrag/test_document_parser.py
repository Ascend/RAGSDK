#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=consider-using-with
"""Unit tests for mx_rag.graphrag.multimodal.parser.DocumentParser."""

import io
import os
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, patch

import requests

from paddle.base import libpaddle  # noqa: F401
from mx_rag.graphrag.multimodal.multimodal_config import MultimodalConfig
from mx_rag.graphrag.multimodal.parser import DocumentParser


def _make_config(**overrides):
    kwargs = dict(
        parser_server="http://127.0.0.1:8000",
        vlm_servers=[["http://127.0.0.1:30000"]],
        vlm_model_name="vlm",
        batch_size=4,
        num_workers_per_server=2,
    )
    kwargs.update(overrides)
    return MultimodalConfig(**kwargs)


def _make_zip(entries):
    """Build an in-memory zip with entries [(name, content), ...]."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in entries:
            zf.writestr(name, content)
    return buf.getvalue()


class _BaseParserTest(unittest.TestCase):
    def setUp(self):
        self._sec_patcher = patch("mx_rag.graphrag.multimodal.multimodal_config.SecDirCheck")
        self._sec_patcher.start()
        self.addCleanup(self._sec_patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._output_folder = self._tmp.name
        self._config = _make_config(output_folder=self._output_folder)
        self._parser = DocumentParser(self._config)

    def _make_file(self, name="doc.pdf", content=b"%PDF-1.4 fake"):
        path = os.path.join(self._tmp.name, name)
        with open(path, "wb") as f:
            f.write(content)
        return path


class TestParserInit(_BaseParserTest):
    def test_stores_config_and_creates_output_folder(self):
        self.assertEqual(self._parser._config, self._config)
        self.assertEqual(self._parser.parser_server, "http://127.0.0.1:8000")
        self.assertTrue(os.path.isdir(self._output_folder))


class TestParse(_BaseParserTest):
    def test_no_valid_files_returns_empty(self):
        md, img = self._parser.parse(["/no/such.pdf"])
        self.assertEqual(md, [])
        self.assertEqual(img, [])

    def test_happy_path_returns_collected_files(self):
        pdf = self._make_file()
        # simulate parser output by pre-placing md/image files
        with open(os.path.join(self._output_folder, "out.md"), "w", encoding="utf-8") as f:
            f.write("# hi")
        with open(os.path.join(self._output_folder, "pic.jpg"), "wb") as f:
            f.write(b"x")
        with patch.object(self._parser, "_batch_infer") as mock_bi:
            md, img = self._parser.parse([pdf])
        mock_bi.assert_called_once_with([pdf])
        self.assertTrue(any(p.endswith("out.md") for p in md))
        self.assertTrue(any(p.endswith("pic.jpg") for p in img))


class TestValidateFiles(_BaseParserTest):
    def test_skips_missing_files(self):
        real = self._make_file()
        valid = self._parser._validate_files([real, "/missing.pdf"])
        self.assertEqual(valid, [real])


class TestCollectFiles(_BaseParserTest):
    def test_collect_markdown_files(self):
        sub = os.path.join(self._output_folder, "sub")
        os.makedirs(sub, exist_ok=True)
        for name in ["a.md", "b.md"]:
            with open(os.path.join(sub, name), "w", encoding="utf-8") as f:
                f.write("x")
        self.assertEqual(len(self._parser._collect_markdown_files()), 2)

    def test_collect_image_files(self):
        sub = os.path.join(self._output_folder, "sub")
        os.makedirs(sub, exist_ok=True)
        # jpg/jpeg only: avoid *.png/*.PNG double-count on case-insensitive FS
        for name in ["a.jpg", "b.jpeg", "c.txt"]:
            with open(os.path.join(sub, name), "wb") as f:
                f.write(b"x")
        self.assertEqual(len(self._parser._collect_image_files()), 2)


class TestPartitionData(_BaseParserTest):
    def test_zero_workers_returns_empty(self):
        self._parser._K = 0
        self.assertEqual(self._parser._partition_data([1, 2, 3]), [])

    def test_partitions_evenly(self):
        self._parser._K = 2
        result = self._parser._partition_data([1, 2, 3, 4, 5])
        self.assertEqual(result, [[1, 2, 3], [4, 5]])

    def test_pads_when_inputs_fewer_than_workers(self):
        self._parser._K = 3
        result = self._parser._partition_data([1, 2])
        self.assertEqual(result, [[1], [2], []])


class TestBatchGenerator(unittest.TestCase):
    def test_chunks(self):
        chunks = list(DocumentParser._batch_generator([1, 2, 3, 4, 5], 2))
        self.assertEqual(chunks, [[1, 2], [3, 4], [5]])

    def test_empty(self):
        self.assertEqual(list(DocumentParser._batch_generator([], 3)), [])


class TestProgressUpdate(_BaseParserTest):
    def test_drains_queue_until_total(self):
        class FakeQueue:
            def __init__(self, items):
                self.items = list(items)

            def empty(self):
                return len(self.items) == 0

            def get(self):
                return self.items.pop(0)

        q = FakeQueue([1, 1])
        # should return once completed (2) == total (2)
        self._parser._progress_update(q, 2)  # must not hang


class TestServerInference(_BaseParserTest):
    def _resp(self, json_data=None, content=b""):
        r = MagicMock()
        r.json.return_value = json_data or {}
        r.raise_for_status.return_value = None
        r.content = content
        return r

    @patch("mx_rag.graphrag.multimodal.parser.time.sleep")
    def test_happy_path_extracts_zip(self, _sleep):
        pdf = self._make_file()
        zip_bytes = _make_zip([("result.md", "# hello")])
        post_resp = self._resp(json_data={"task_id": "t1"})
        completed_resp = self._resp(json_data={"status": "completed"})
        result_resp = self._resp(content=zip_bytes)
        queue = MagicMock()
        with (
            patch("mx_rag.graphrag.multimodal.parser.requests.post", return_value=post_resp),
            patch("mx_rag.graphrag.multimodal.parser.requests.get", side_effect=[completed_resp, result_resp]),
        ):
            self._parser._server_inference("http://srv", [pdf], queue)
        self.assertTrue(os.path.isfile(os.path.join(self._output_folder, "result.md")))
        queue.put.assert_called_with(1)

    @patch("mx_rag.graphrag.multimodal.parser.time.sleep")
    def test_polls_then_completes(self, mock_sleep):
        pdf = self._make_file()
        zip_bytes = _make_zip([("r.md", "x")])
        post_resp = self._resp(json_data={"task_id": "t1"})
        running_resp = self._resp(json_data={"status": "running"})
        completed_resp = self._resp(json_data={"status": "completed"})
        result_resp = self._resp(content=zip_bytes)
        queue = MagicMock()
        with (
            patch("mx_rag.graphrag.multimodal.parser.requests.post", return_value=post_resp),
            patch(
                "mx_rag.graphrag.multimodal.parser.requests.get",
                side_effect=[running_resp, completed_resp, result_resp],
            ),
        ):
            self._parser._server_inference("http://srv", [pdf], queue)
        mock_sleep.assert_called_once_with(2)

    @patch("mx_rag.graphrag.multimodal.parser.time.sleep")
    def test_no_task_id_continues(self, _sleep):
        pdf = self._make_file()
        post_resp = self._resp(json_data={})  # no task_id
        queue = MagicMock()
        with (
            patch("mx_rag.graphrag.multimodal.parser.requests.post", return_value=post_resp),
            patch("mx_rag.graphrag.multimodal.parser.requests.get") as mock_get,
        ):
            self._parser._server_inference("http://srv", [pdf], queue)
        mock_get.assert_not_called()
        queue.put.assert_called_with(1)

    @patch("mx_rag.graphrag.multimodal.parser.time.sleep")
    def test_post_request_exception_caught(self, _sleep):
        pdf = self._make_file()
        queue = MagicMock()
        with (
            patch("mx_rag.graphrag.multimodal.parser.requests.post", side_effect=requests.RequestException("boom")),
            patch("mx_rag.graphrag.multimodal.parser.requests.get") as mock_get,
        ):
            self._parser._server_inference("http://srv", [pdf], queue)
        mock_get.assert_not_called()
        queue.put.assert_called_with(1)

    @patch("mx_rag.graphrag.multimodal.parser.time.sleep")
    def test_post_generic_exception_caught(self, _sleep):
        pdf = self._make_file()
        queue = MagicMock()
        with (
            patch("mx_rag.graphrag.multimodal.parser.requests.post", side_effect=ValueError("boom")),
            patch("mx_rag.graphrag.multimodal.parser.requests.get") as mock_get,
        ):
            self._parser._server_inference("http://srv", [pdf], queue)
        mock_get.assert_not_called()
        queue.put.assert_called_with(1)

    @patch("mx_rag.graphrag.multimodal.parser.time.sleep")
    def test_failed_status_branch(self, _sleep):
        pdf = self._make_file()
        post_resp = self._resp(json_data={"task_id": "t1"})
        failed_resp = self._resp(json_data={"status": "failed", "error": "oops"})
        # after failed -> break -> proceeds to result download -> raise RequestException
        result_resp = MagicMock()
        result_resp.raise_for_status.side_effect = requests.RequestException("no result")
        queue = MagicMock()
        with (
            patch("mx_rag.graphrag.multimodal.parser.requests.post", return_value=post_resp),
            patch("mx_rag.graphrag.multimodal.parser.requests.get", side_effect=[failed_resp, result_resp]),
        ):
            self._parser._server_inference("http://srv", [pdf], queue)
        # progress put invoked (failed branch + trailing)
        self.assertGreaterEqual(queue.put.call_count, 1)


class TestBatchInfer(_BaseParserTest):
    def test_dispatches_partitions_to_pool(self):
        pdf = self._make_file()
        self._parser._K = 2
        pool_mock = MagicMock()
        pool_mock.__enter__.return_value = pool_mock
        mp_mock = MagicMock()
        mp_mock.Pool.return_value = pool_mock
        with (
            patch("mx_rag.graphrag.multimodal.parser.multiprocessing", mp_mock),
            patch.object(self._parser, "_partition_data", return_value=[[pdf], [pdf]]),
            patch.object(self._parser, "_server_inference"),
            patch.object(self._parser, "_progress_update"),
        ):
            self._parser._batch_infer([pdf, pdf])
        self.assertEqual(pool_mock.apply_async.call_count, 2)
        mp_mock.Process.return_value.start.assert_called_once()
        mp_mock.Process.return_value.join.assert_called_once()


if __name__ == "__main__":
    unittest.main()
