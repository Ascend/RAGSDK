#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for mx_rag.graphrag.multimodal.vlm_inference.VLMInferenceEngine."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from paddle.base import libpaddle  # noqa: F401
from mx_rag.graphrag.multimodal.multimodal_config import MultimodalConfig
from mx_rag.graphrag.multimodal.vlm_inference import VLMInferenceEngine


def _make_config(**overrides):
    kwargs = dict(
        parser_server="http://127.0.0.1:8000",
        vlm_servers=[["http://127.0.0.1:30000"]],
        vlm_model_name="vlm",
        num_workers_per_server=2,
    )
    kwargs.update(overrides)
    return MultimodalConfig(**kwargs)


class TestVLMInit(unittest.TestCase):
    def setUp(self):
        patcher = patch("mx_rag.graphrag.multimodal.multimodal_config.SecDirCheck")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_service_mode_init_pool(self):
        with patch("mx_rag.graphrag.multimodal.vlm_inference.Img2TextLLM") as mock_llm:
            engine = VLMInferenceEngine(_make_config(vlm_servers=[["http://s1"], ["http://s2"]]))
        self.assertEqual(len(engine._SERVERS), 2)
        # K = min(num_workers_per_server=2, 64) = 2 -> each round replicated 2x
        self.assertEqual(len(engine._SERVERS[0]), 2)
        self.assertEqual(len(engine._llm_pool[0]), 2)
        self.assertEqual(mock_llm.call_count, 4)

    def test_no_vlm_servers_raises_value_error(self):
        with self.assertRaises(ValueError):
            VLMInferenceEngine(_make_config(vlm_servers=[]))


class TestVLMRun(unittest.TestCase):
    def setUp(self):
        patcher = patch("mx_rag.graphrag.multimodal.multimodal_config.SecDirCheck")
        patcher.start()
        self.addCleanup(patcher.stop)
        with patch("mx_rag.graphrag.multimodal.vlm_inference.Img2TextLLM"):
            self.engine = VLMInferenceEngine(_make_config())

    def test_run_calls_infer_service(self):
        with patch.object(self.engine, "_infer_service", new=AsyncMock(return_value=["srv"])) as mock_svc:
            result = asyncio.run(self.engine.run([["p", "i"]], idx=0))
        self.assertEqual(result, ["srv"])
        mock_svc.assert_called_once()


class TestInferService(unittest.TestCase):
    def setUp(self):
        patcher = patch("mx_rag.graphrag.multimodal.multimodal_config.SecDirCheck")
        patcher.start()
        self.addCleanup(patcher.stop)
        with patch("mx_rag.graphrag.multimodal.vlm_inference.Img2TextLLM"):
            self.engine = VLMInferenceEngine(_make_config())

    def test_idx_out_of_range_returns_empty_strings(self):
        batch = [["p", "i"], ["p2", "i2"]]
        result = asyncio.run(self.engine._infer_service(batch, idx=99))
        self.assertEqual(result, ["", ""])

    def test_happy_path_dispatches_to_call_service(self):
        self.engine._SERVERS = [["s1"]]
        self.engine._llm_pool = [[MagicMock()]]
        batch = [["promptA", "imgA"], ["promptB", "imgB"]]

        def fake_call(llm, prompt, image):
            return f"{prompt}|{image}"

        with patch.object(VLMInferenceEngine, "_call_service", side_effect=fake_call):
            result = asyncio.run(self.engine._infer_service(batch, idx=0))
        self.assertEqual(result, ["promptA|imgA", "promptB|imgB"])

    def test_non_string_response_coerced_to_empty(self):
        self.engine._SERVERS = [["s1"]]
        self.engine._llm_pool = [[MagicMock()]]
        with patch.object(VLMInferenceEngine, "_call_service", return_value=None):
            result = asyncio.run(self.engine._infer_service([["p", "i"]], idx=0))
        self.assertEqual(result, [""])


class TestCallService(unittest.TestCase):
    def test_happy_path(self):
        llm = MagicMock()
        llm.chat.return_value = "caption"
        with patch("mx_rag.graphrag.multimodal.vlm_inference.OpenAIEmbedding") as mock_emb:
            mock_emb.encode_image_to_base64.return_value = "data:image/png;base64,AAA"
            result = VLMInferenceEngine._call_service(llm, "prompt", "img.png")
        self.assertEqual(result, "caption")
        llm.chat.assert_called_once()
        _, kwargs = llm.chat.call_args
        self.assertEqual(kwargs["image_url"], {"url": "data:image/png;base64,AAA"})
        self.assertEqual(kwargs["prompt"], "prompt")

    def test_exception_returns_empty_string(self):
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("boom")
        with patch("mx_rag.graphrag.multimodal.vlm_inference.OpenAIEmbedding") as mock_emb:
            mock_emb.encode_image_to_base64.return_value = "data:image/png;base64,AAA"
            result = VLMInferenceEngine._call_service(llm, "prompt", "img.png")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
