#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""Unit tests for mx_rag.graphrag.multimodal.verifier.CaptionVerifier."""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from mx_rag.graphrag.multimodal.multimodal_config import MultimodalConfig
from mx_rag.graphrag.multimodal.verifier import CaptionVerifier


def _make_config(**overrides):
    kwargs = dict(
        parser_server="http://127.0.0.1:8000",
        vlm_servers=[["http://127.0.0.1:30000"]],
        vlm_model_name="vlm",
        emb_server_url="http://emb/v1",
        emb_model_name="emb",
    )
    kwargs.update(overrides)
    return MultimodalConfig(**kwargs)


class TestCaptionVerifierInit(unittest.TestCase):
    def setUp(self):
        patcher = patch("mx_rag.graphrag.multimodal.multimodal_config.SecDirCheck")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_service_mode_creates_openai_embed(self):
        with patch("mx_rag.graphrag.multimodal.verifier.OpenAIEmbedding") as mock_emb:
            verifier = CaptionVerifier(_make_config(emb_server_url="http://emb/v1"))
            mock_emb.assert_called_once_with(model_name="emb", url="http://emb/v1")
            self.assertIsNone(verifier._model)
            self.assertIsNotNone(verifier._openai_embed)

    def test_no_embed_configured(self):
        verifier = CaptionVerifier(_make_config(emb_server_url=None, emb_model_path=None, emb_model_name=None))
        self.assertIsNone(verifier._openai_embed)
        self.assertIsNone(verifier._model)

    def test_local_mode_load_model_import_error_raises_runtime(self):
        # Setting sys.modules[name] = None forces ImportError on `from transformers import ...`.
        with patch.dict(sys.modules, {"transformers": None, "torch": None}):
            with self.assertRaises(RuntimeError):
                CaptionVerifier(_make_config(emb_server_url=None, emb_model_path="/data/emb"))

    def test_local_mode_load_model_success(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "fp16"
        fake_model = MagicMock()
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModel = MagicMock()
        fake_transformers.AutoModel.from_pretrained.return_value = fake_model
        with patch.dict(sys.modules, {"transformers": fake_transformers, "torch": fake_torch}):
            verifier = CaptionVerifier(_make_config(emb_server_url=None, emb_model_path="/data/emb"))
        self.assertIs(verifier._model, fake_model)
        fake_model.to.assert_called_once_with("npu:0")


class TestCaptionVerifierVerify(unittest.TestCase):
    def setUp(self):
        patcher = patch("mx_rag.graphrag.multimodal.multimodal_config.SecDirCheck")
        patcher.start()
        self.addCleanup(patcher.stop)
        with patch("mx_rag.graphrag.multimodal.verifier.OpenAIEmbedding"):
            self.verifier = CaptionVerifier(_make_config())
        # defaults: truncate_dim=512, filter_type=4

    def test_verify_skipped_when_no_embed(self):
        verifier = CaptionVerifier(_make_config(emb_server_url=None, emb_model_path=None, emb_model_name=None))
        captions = [{"id": "a.png", "text": "hi"}]
        self.assertIs(verifier.verify(captions), captions)

    def test_verify_empty_captions(self):
        self.assertEqual(self.verifier.verify([]), [])

    def test_verify_processes_each_caption(self):
        captions = [{"id": "a.png", "text": "old1"}, {"id": "b.png", "text": "old2"}]
        with patch.object(self.verifier, "process_text", side_effect=["new1", "new2"]) as mock_pt:
            result = self.verifier.verify(captions)
        self.assertEqual([c["text"] for c in result], ["new1", "new2"])
        mock_pt.assert_any_call("a.png", "old1")
        mock_pt.assert_any_call("b.png", "old2")

    def test_verify_handles_process_text_exception(self):
        captions = [{"id": "a.png", "text": "old"}]
        with patch.object(self.verifier, "process_text", side_effect=ValueError("boom")):
            result = self.verifier.verify(captions)
        # original text preserved when processing raises
        self.assertEqual(result[0]["text"], "old")


class TestProcessText(unittest.TestCase):
    def setUp(self):
        patcher = patch("mx_rag.graphrag.multimodal.multimodal_config.SecDirCheck")
        patcher.start()
        self.addCleanup(patcher.stop)
        with patch("mx_rag.graphrag.multimodal.verifier.OpenAIEmbedding"):
            self.verifier = CaptionVerifier(_make_config(filter_type=2))

    def test_empty_text_returns_empty(self):
        self.assertEqual(self.verifier.process_text("a.png", ""), "")
        self.assertEqual(self.verifier.process_text("a.png", "   "), "")

    def test_chunks_all_empty_returns_empty(self):
        # "。" splitting yields whitespace-only chunks that collapse to empty.
        self.assertEqual(self.verifier.process_text("a.png", " 。 。 "), "")

    def test_similarity_empty_returns_joined_chunks(self):
        with patch.object(self.verifier, "_compute_similarity", return_value=np.array([])):
            result = self.verifier.process_text("a.png", "甲。乙。丙")
        self.assertEqual(result, "甲。乙。丙")

    def test_all_filtered_out_returns_joined_chunks(self):
        with (
            patch.object(self.verifier, "_compute_similarity", return_value=np.array([0.1, 0.2])),
            patch.object(self.verifier, "_filter_option", return_value=999.0),
        ):
            result = self.verifier.process_text("a.png", "甲。乙")
        self.assertEqual(result, "甲。乙")

    def test_normal_filtering_keeps_above_threshold(self):
        with (
            patch.object(self.verifier, "_compute_similarity", return_value=np.array([0.1, 0.9, 0.5])),
            patch.object(self.verifier, "_filter_option", return_value=0.4),
        ):
            result = self.verifier.process_text("a.png", "甲。乙。丙")
        self.assertEqual(result, "乙。丙")


class TestComputeSimilarity(unittest.TestCase):
    def setUp(self):
        patcher = patch("mx_rag.graphrag.multimodal.multimodal_config.SecDirCheck")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _build(self, truncate_dim=512):
        with patch("mx_rag.graphrag.multimodal.verifier.OpenAIEmbedding"):
            return CaptionVerifier(_make_config(truncate_dim=truncate_dim))

    def test_service_path_without_truncation(self):
        verifier = self._build(truncate_dim=512)
        verifier._openai_embed = MagicMock()
        verifier._openai_embed.run_text.return_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        verifier._openai_embed.run_image.return_value = [[1.0, 1.0, 0.0]]
        sim = verifier._compute_similarity_service(["a", "b"], "img.png")
        verifier._openai_embed.run_text.assert_called_once_with(["a", "b"])
        verifier._openai_embed.run_image.assert_called_once_with(["img.png"])
        self.assertEqual(sim.shape, (2,))

    def test_service_path_with_truncation(self):
        verifier = self._build(truncate_dim=2)
        verifier._openai_embed = MagicMock()
        verifier._openai_embed.run_text.return_value = [[1.0, 0.0, 0.0, 0.0]]
        verifier._openai_embed.run_image.return_value = [[1.0, 0.0, 1.0, 1.0]]
        sim = verifier._compute_similarity_service(["a"], "img.png")
        # truncated to first 2 dims: img·text = 1*1 + 0*0 = 1.0
        self.assertAlmostEqual(float(sim[0]), 1.0)

    def test_local_path(self):
        with patch("mx_rag.graphrag.multimodal.multimodal_config.SecDirCheck"):
            verifier = CaptionVerifier(_make_config(emb_server_url=None, emb_model_name=None))
        verifier._model = MagicMock()
        text_emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        img_emb = np.array([[1.0, 1.0]])
        verifier._model.encode_text.return_value = text_emb
        verifier._model.encode_image.return_value = img_emb
        sim = verifier._compute_similarity_local(["a", "b"], "img.png")
        self.assertEqual(sim.shape, (2,))

    def test_compute_similarity_swallows_exception(self):
        verifier = self._build()
        verifier._openai_embed = MagicMock()
        verifier._openai_embed.run_text.side_effect = RuntimeError("net")
        sim = verifier._compute_similarity(["a"], "img.png")
        self.assertEqual(sim.size, 0)


class TestFilterOptions(unittest.TestCase):
    def _verifier(self):
        with (
            patch("mx_rag.graphrag.multimodal.multimodal_config.SecDirCheck"),
            patch("mx_rag.graphrag.multimodal.verifier.OpenAIEmbedding"),
        ):
            return CaptionVerifier(_make_config(emb_server_url=None, emb_model_path=None, emb_model_name=None))

    def test_filter_option_dispatch(self):
        sims = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        verifier = self._verifier()
        with patch.object(CaptionVerifier, "_filter_largest_percent", return_value=0.1) as m1:
            self.assertEqual(verifier._filter_option(sims, 1), 0.1)
            m1.assert_called_once()
        with patch.object(CaptionVerifier, "_filter_by_std", return_value=0.2):
            self.assertEqual(verifier._filter_option(sims, 2), 0.2)
        with patch.object(CaptionVerifier, "_filter_by_max_delta_segment", return_value=0.3):
            self.assertEqual(verifier._filter_option(sims, 3), 0.3)
        with patch.object(CaptionVerifier, "_filter_by_max_delta_segment_with_std", return_value=0.4):
            self.assertEqual(verifier._filter_option(sims, 4), 0.4)
        self.assertEqual(verifier._filter_option(sims, 99), 0.0)

    def test_filter_largest_percent(self):
        sims = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        self.assertAlmostEqual(CaptionVerifier._filter_largest_percent(sims), 0.3)
        # n_remove == 0 returns 0.0
        self.assertEqual(CaptionVerifier._filter_largest_percent(np.array([0.5, 0.6]), percent=20), 0.0)

    def test_filter_by_std(self):
        sims = np.array([0.1, 0.3, 0.5])
        threshold = CaptionVerifier._filter_by_std(sims)
        self.assertGreaterEqual(threshold, 0.1)
        self.assertLess(threshold, 0.5)

    def test_filter_by_max_delta_segment_short_returns_zero(self):
        self.assertEqual(CaptionVerifier._filter_by_max_delta_segment(np.array([0.1, 0.2])), 0.0)

    def test_filter_by_max_delta_segment_length_3_returns_zero(self):
        # length=3 with default low=0.1, high=0.3 -> segment too short
        sims = np.array([0.1, 0.5, 0.9])
        self.assertEqual(CaptionVerifier._filter_by_max_delta_segment(sims), 0.0)

    def test_filter_by_max_delta_segment_normal(self):
        sims = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        threshold = CaptionVerifier._filter_by_max_delta_segment(sims)
        self.assertIn(threshold, [0.1, 0.3, 0.5, 0.7, 0.9])

    def test_filter_by_max_delta_segment_with_std_short_returns_zero(self):
        self.assertEqual(CaptionVerifier._filter_by_max_delta_segment_with_std(np.array([0.1, 0.9])), 0.0)

    def test_filter_by_max_delta_segment_with_std_length_3_returns_zero(self):
        # length=3 with default low=0.1, high=0.3 -> segment too short
        sims = np.array([0.1, 0.5, 0.9])
        self.assertEqual(CaptionVerifier._filter_by_max_delta_segment_with_std(sims), 0.0)

    def test_filter_by_max_delta_segment_with_std_normal(self):
        sims = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        threshold = CaptionVerifier._filter_by_max_delta_segment_with_std(sims)
        self.assertGreaterEqual(threshold, 0.0)


class TestChunkText(unittest.TestCase):
    def test_split_by_chinese_period(self):
        self.assertEqual(CaptionVerifier.chunk_text("甲。乙。丙"), ["甲", "乙", "丙"])

    def test_split_by_dot_when_no_chinese_period(self):
        self.assertEqual(CaptionVerifier.chunk_text("a.b.c"), ["a", "b", "c"])

    def test_mixed_uses_chinese_period(self):
        self.assertEqual(CaptionVerifier.chunk_text("甲。b.c"), ["甲", "b.c"])


if __name__ == "__main__":
    unittest.main()
