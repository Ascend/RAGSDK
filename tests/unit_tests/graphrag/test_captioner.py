#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=consider-using-with
"""Unit tests for mx_rag.graphrag.multimodal.captioner.ImageCaptioner."""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from paddle.base import libpaddle  # noqa: F401
from mx_rag.graphrag.multimodal.captioner import ImageCaptioner
from mx_rag.graphrag.multimodal.multimodal_config import MultimodalConfig
from mx_rag.graphrag.prompts.multimodal_prompt import (
    MULTIMODAL_FINE_PROMPT_CN,
    MULTIMODAL_INIT_PROMPT_CN,
)


def _make_config(**overrides):
    kwargs = dict(
        parser_server="http://127.0.0.1:8000",
        vlm_servers=[["http://127.0.0.1:30000"]],
        vlm_model_name="vlm",
        batch_size=64,
        num_workers_per_server=2,
    )
    kwargs.update(overrides)
    return MultimodalConfig(**kwargs)


class _BaseCaptionerTest(unittest.TestCase):
    def setUp(self):
        self._sec_patcher = patch("mx_rag.graphrag.multimodal.multimodal_config.SecDirCheck")
        self._sec_patcher.start()
        self.addCleanup(self._sec_patcher.stop)
        self._vlm_patcher = patch("mx_rag.graphrag.multimodal.captioner.VLMInferenceEngine")
        self._vlm_cls = self._vlm_patcher.start()
        self.addCleanup(self._vlm_patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _build(self, **overrides):
        overrides.setdefault("output_folder", os.path.join(self._tmp.name, "output"))
        captioner = ImageCaptioner(_make_config(**overrides))
        captioner.set_vlm_result_dir(self._tmp.name)
        return captioner


class TestLoadPrompts(_BaseCaptionerTest):
    def test_default_built_in_prompts(self):
        captioner = self._build()
        self.assertEqual(captioner._prompts_dict["init_prompt"], MULTIMODAL_INIT_PROMPT_CN)
        self.assertEqual(captioner._prompts_dict["fine_prompt"], MULTIMODAL_FINE_PROMPT_CN)

    def test_load_prompts_from_json_file(self):
        prompt_file = os.path.join(self._tmp.name, "prompts.json")
        with open(prompt_file, "w", encoding="utf-8") as f:
            json.dump({"init_prompt": "I", "fine_prompt": "F"}, f)
        captioner = self._build(prompt_path=prompt_file)
        self.assertEqual(captioner._prompts_dict, {"init_prompt": "I", "fine_prompt": "F"})

    def test_invalid_prompt_path_falls_back_to_builtin(self):
        captioner = self._build(prompt_path=os.path.join(self._tmp.name, "missing.json"))
        self.assertEqual(captioner._prompts_dict["init_prompt"], MULTIMODAL_INIT_PROMPT_CN)


class TestSetVlmResultDir(_BaseCaptionerTest):
    def test_creates_directory(self):
        captioner = self._build()
        new_dir = os.path.join(self._tmp.name, "nested", "results")
        captioner.set_vlm_result_dir(new_dir)
        self.assertEqual(captioner._vlm_result_dir, new_dir)
        self.assertTrue(os.path.isdir(new_dir))


class TestCaption(_BaseCaptionerTest):
    def test_empty_image_list_returns_empty(self):
        captioner = self._build()
        self.assertEqual(asyncio.run(captioner.caption([])), [])

    def test_no_valid_images_returns_empty(self):
        captioner = self._build()
        self.assertEqual(asyncio.run(captioner.caption(["a.txt", "b.doc"])), [])

    def test_single_loop_happy_path(self):
        captioner = self._build()  # _loops = 1
        captioner._generator.run = AsyncMock(return_value=["cap1", "cap2"])
        result = asyncio.run(captioner.caption(["img1.jpg", "img2.png"]))
        self.assertEqual(len(result), 2)
        ids = [c["id"] for c in result]
        texts = [c["text"] for c in result]
        self.assertIn("img1.jpg", ids)
        self.assertIn("img2.png", ids)
        self.assertEqual(set(texts), {"cap1", "cap2"})
        # result file persisted
        result_path = os.path.join(self._tmp.name, "tmp.json")
        self.assertTrue(os.path.isfile(result_path))

    def test_multi_loop_appends_fine_results(self):
        captioner = self._build(vlm_servers=[["http://s1"], ["http://s2"]])  # _loops = 2
        captioner._generator.run = AsyncMock(side_effect=[["t1", "t2"], ["f1", "f2"]])
        result = asyncio.run(captioner.caption(["img1.jpg", "img2.png"]))
        # only init-round texts are written (fine results extend generate_texts beyond len(batch))
        self.assertEqual([c["text"] for c in result], ["t1", "t2"])
        self.assertEqual(captioner._generator.run.await_count, 2)

    def test_non_string_result_coerced(self):
        captioner = self._build()
        captioner._generator.run = AsyncMock(return_value=[None, 123])
        result = asyncio.run(captioner.caption(["img1.jpg", "img2.png"]))
        # None -> "" (written but dropped on reload since text is empty); 123 -> "123"
        texts = [c["text"] for c in result]
        self.assertEqual(texts, ["123"])


class TestFilterImageFiles(_BaseCaptionerTest):
    def test_filters_by_extension(self):
        captioner = self._build()
        filtered = captioner._filter_image_files(["a.jpg", "b.txt", "c.PNG", "noext"])
        self.assertEqual(filtered, ["a.jpg", "c.PNG"])


class TestLoadCaptions(_BaseCaptionerTest):
    def test_file_not_exist_returns_empty(self):
        self.assertEqual(ImageCaptioner._load_captions("/no/such/file.json"), [])

    def test_parses_valid_lines_and_skips_others(self):
        path = os.path.join(self._tmp.name, "caps.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "a", "text": "t1"}) + "\n")
            f.write("\n")
            f.write("not json\n")
            f.write(json.dumps({"id": "b", "text": ""}) + "\n")  # no text -> skipped
            f.write(json.dumps({"id": "c", "text": "t3"}) + "\n")
        captions = ImageCaptioner._load_captions(path)
        self.assertEqual(len(captions), 2)
        self.assertEqual([c["id"] for c in captions], ["a", "c"])


class TestBatchGenerator(unittest.TestCase):
    def test_batches(self):
        batches = list(ImageCaptioner._batch_generator([1, 2, 3, 4, 5], 2))
        self.assertEqual(batches, [[1, 2], [3, 4], [5]])

    def test_empty(self):
        self.assertEqual(list(ImageCaptioner._batch_generator([], 3)), [])


class TestReplaceImageMarkdown(unittest.TestCase):
    def test_removes_markdown_images(self):
        text = "before ![](http://x/a.png) middle ![alt](b.png) end"
        self.assertEqual(ImageCaptioner._replace_image_markdown(text), "before  middle  end")

    def test_no_markdown_unchanged(self):
        self.assertEqual(ImageCaptioner._replace_image_markdown("plain text"), "plain text")


class TestBuildFineInputs(_BaseCaptionerTest):
    def test_builds_fine_inputs_for_valid_images(self):
        captioner = self._build()
        batch = [["init_prompt", "img1.jpg"], ["init_prompt", "img2.png"]]
        generate_texts = ["caption1", "caption2"]
        result = captioner._build_fine_inputs(batch, generate_texts)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], "caption1" + captioner._prompts_dict["fine_prompt"])
        self.assertEqual(result[0][1], "img1.jpg")

    def test_skips_non_image_files(self):
        captioner = self._build()
        batch = [["init_prompt", "img1.jpg"], ["init_prompt", "doc.txt"]]
        generate_texts = ["caption1", "caption2"]
        result = captioner._build_fine_inputs(batch, generate_texts)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "img1.jpg")

    def test_handles_none_generate_texts(self):
        captioner = self._build()
        batch = [["init_prompt", "img1.jpg"]]
        generate_texts = [None]
        result = captioner._build_fine_inputs(batch, generate_texts)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0][0].endswith(captioner._prompts_dict["fine_prompt"]))


class TestWriteBatchResults(_BaseCaptionerTest):
    def test_writes_results_to_file(self):
        import io

        captioner = self._build()
        f = io.StringIO()
        batch = [["init_prompt", "img1.jpg"], ["init_prompt", "img2.png"]]
        generate_texts = ["caption1", "caption2"]
        captioner._write_batch_results(f, batch, generate_texts)
        lines = f.getvalue().strip().split("\n")
        self.assertEqual(len(lines), 2)
        data1 = json.loads(lines[0])
        self.assertEqual(data1["id"], "img1.jpg")
        self.assertEqual(data1["text"], "caption1")

    def test_stops_at_batch_boundary(self):
        import io

        captioner = self._build()
        f = io.StringIO()
        batch = [["init_prompt", "img1.jpg"]]
        generate_texts = ["cap1", "cap2", "cap3"]  # more than batch
        captioner._write_batch_results(f, batch, generate_texts)
        lines = f.getvalue().strip().split("\n")
        self.assertEqual(len(lines), 1)  # only 1 item in batch


if __name__ == "__main__":
    unittest.main()
