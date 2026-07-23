#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for mx_rag.graphrag.multimodal.multimodal_config.MultimodalConfig."""

import tempfile
import unittest
from unittest.mock import patch

from paddle.base import libpaddle  # noqa: F401
from mx_rag.graphrag.multimodal.multimodal_config import (
    MAX_FILE_SIZE_10M,
    SUPPORTED_IMAGE_EXTENSIONS,
    MultimodalConfig,
)


class TestMultimodalConfig(unittest.TestCase):
    """Cover MultimodalConfig construction and all __post_init__ validations."""

    def setUp(self):
        # SecDirCheck performs filesystem/owner validation that is environment
        # specific (requires POSIX absolute paths + os.getuid); it is not the
        # unit under test here, so neutralise it during config construction.
        patcher = patch("mx_rag.graphrag.multimodal.multimodal_config.SecDirCheck")
        self._sec_mock = patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self._tmp.cleanup)
        self._output_folder = self._tmp.name

    def _base_kwargs(self, **overrides):
        kwargs = dict(
            parser_server="http://127.0.0.1:8000",
            vlm_servers=[["http://127.0.0.1:30000"]],
            vlm_model_name="test-vlm",
            output_folder=self._output_folder,
        )
        kwargs.update(overrides)
        return kwargs

    def test_valid_config_service_mode(self):
        cfg = MultimodalConfig(**self._base_kwargs())
        self.assertEqual(cfg.parser_server, "http://127.0.0.1:8000")
        self.assertEqual(cfg.vlm_model_name, "test-vlm")
        self.assertEqual(cfg.merge_type, "replace")
        self.assertEqual(cfg.filter_type, 4)
        self.assertEqual(cfg.batch_size, 64)
        self.assertEqual(cfg.num_workers_per_server, 8)
        self.assertEqual(cfg.timeout, 300)
        self.assertEqual(cfg.device, "npu:0")
        # SecDirCheck invoked once with the configured output folder.
        self._sec_mock.assert_called_once_with(self._output_folder, MAX_FILE_SIZE_10M)

    def test_custom_optional_fields(self):
        cfg = MultimodalConfig(
            **self._base_kwargs(
                emb_server_url="http://emb/v1",
                emb_model_name="emb-model",
                filter_type=2,
                truncate_dim=128,
                num_workers_per_server=4,
                batch_size=8,
                merge_type="append",
                vlm_result_name="result",
                device="cpu",
                timeout=120,
            )
        )
        self.assertEqual(cfg.filter_type, 2)
        self.assertEqual(cfg.truncate_dim, 128)
        self.assertEqual(cfg.merge_type, "append")
        self.assertEqual(cfg.device, "cpu")

    def test_default_output_folder_created_when_not_provided(self):
        with patch("mx_rag.graphrag.multimodal.multimodal_config.os.makedirs") as makedirs:
            cfg = MultimodalConfig(
                parser_server="http://127.0.0.1:8000",
                vlm_servers=[["http://127.0.0.1:30000"]],
                vlm_model_name="test-vlm",
            )
        self.assertTrue(cfg.output_folder)
        makedirs.assert_called_once_with(cfg.output_folder, 0o750, exist_ok=True)

    def test_invalid_parser_server_empty(self):
        with self.assertRaises(ValueError):
            MultimodalConfig(**self._base_kwargs(parser_server=""))

    def test_invalid_no_vlm_servers(self):
        with self.assertRaises(ValueError):
            MultimodalConfig(**self._base_kwargs(vlm_servers=[]))

    def test_invalid_vlm_model_name_empty(self):
        with self.assertRaises(ValueError):
            MultimodalConfig(**self._base_kwargs(vlm_model_name=""))

    def test_invalid_merge_type(self):
        with self.assertRaises(ValueError):
            MultimodalConfig(**self._base_kwargs(merge_type="overwrite"))

    def test_invalid_filter_type(self):
        with self.assertRaises(ValueError):
            MultimodalConfig(**self._base_kwargs(filter_type=5))

    def test_invalid_batch_size_zero(self):
        with self.assertRaises(ValueError):
            MultimodalConfig(**self._base_kwargs(batch_size=0))

    def test_invalid_num_workers_zero(self):
        with self.assertRaises(ValueError):
            MultimodalConfig(**self._base_kwargs(num_workers_per_server=0))

    def test_constants(self):
        self.assertEqual(SUPPORTED_IMAGE_EXTENSIONS, ["jpg", "jpeg", "png", "PNG"])
        self.assertEqual(MAX_FILE_SIZE_10M, 10 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
