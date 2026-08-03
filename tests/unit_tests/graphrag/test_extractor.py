#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""Unit tests for mx_rag.graphrag.multimodal.extractor.MultimodalPrepare."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from mx_rag.graphrag.multimodal.extractor import MultimodalPrepare
from mx_rag.graphrag.multimodal.multimodal_config import MultimodalConfig


def _make_config(**overrides):
    kwargs = dict(
        parser_server="http://127.0.0.1:8000",
        vlm_servers=[["http://127.0.0.1:30000"]],
        vlm_model_name="vlm",
    )
    kwargs.update(overrides)
    return MultimodalConfig(**kwargs)


class _BaseExtractorTest(unittest.TestCase):
    """Build a MultimodalPrepare with parser/captioner/verifier replaced by mocks."""

    def setUp(self):
        self._patches = [
            patch("mx_rag.graphrag.multimodal.multimodal_config.SecDirCheck"),
            patch("mx_rag.graphrag.multimodal.extractor.DocumentParser"),
            patch("mx_rag.graphrag.multimodal.extractor.ImageCaptioner"),
            patch("mx_rag.graphrag.multimodal.extractor.CaptionVerifier"),
        ]
        self._sec_mock, self._parser_cls, self._captioner_cls, self._verifier_cls = (p.start() for p in self._patches)
        for p in self._patches:
            self.addCleanup(p.stop)
        self._tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self._tmp.cleanup)
        self._output_folder = self._tmp.name
        self._pdf = os.path.join(self._tmp.name, "doc.pdf")
        with open(self._pdf, "wb") as f:
            f.write(b"%PDF-1.4 fake")

    def _build(self, **overrides):
        overrides.setdefault("output_folder", self._output_folder)
        prepare = MultimodalPrepare(_make_config(**overrides))
        return prepare


class TestExtractorInit(_BaseExtractorTest):
    def test_invalid_config_type_raises(self):
        with self.assertRaises(ValueError):
            MultimodalPrepare("not a config")

    def test_verifier_none_without_emb_config(self):
        prepare = self._build()
        self.assertIsNone(prepare._verifier)
        self._verifier_cls.assert_not_called()

    def test_verifier_created_with_emb_server(self):
        prepare = self._build(emb_server_url="http://emb/v1", emb_model_name="emb")
        self.assertIsNotNone(prepare._verifier)
        self._verifier_cls.assert_called_once()

    def test_captioner_receives_vlm_result_dir(self):
        prepare = self._build()
        expected = os.path.join(self._output_folder, "vlm_result")
        prepare._captioner.set_vlm_result_dir.assert_called_once_with(expected)


class TestExtractValidation(_BaseExtractorTest):
    def test_not_a_list_raises(self):
        prepare = self._build()
        with self.assertRaises(ValueError):
            prepare.extract("not a list")

    def test_empty_list_raises(self):
        prepare = self._build()
        with self.assertRaises(ValueError):
            prepare.extract([])

    def test_too_many_files_raises(self):
        prepare = self._build()
        with self.assertRaises(ValueError):
            prepare.extract([self._pdf] * 101)

    def test_missing_file_raises(self):
        prepare = self._build()
        with self.assertRaises(FileNotFoundError):
            prepare.extract(["/no/such/file.pdf"])


class TestExtractFlow(_BaseExtractorTest):
    def test_no_markdown_files_returns_early(self):
        prepare = self._build()
        prepare._parser.parse.return_value = ([], [])
        prepare.extract([self._pdf])
        prepare._captioner.caption.assert_not_called()

    def test_no_images_copies_markdown_directly(self):
        prepare = self._build()
        # place a real markdown file inside output_folder
        sub = os.path.join(self._output_folder, "doc1", "sub")
        os.makedirs(sub, exist_ok=True)
        md_path = os.path.join(sub, "note.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# title")
        prepare._parser.parse.return_value = ([md_path], [])
        prepare.extract([self._pdf])
        copied = os.path.join(self._output_folder, "note.md")
        self.assertTrue(os.path.isfile(copied))

    def test_images_captioning_non_running_loop_branch(self):
        prepare = self._build(emb_server_url="http://emb/v1", emb_model_name="emb")
        prepare._parser.parse.return_value = (["a.md"], ["img1.jpg"])
        captions = [{"id": "img1.jpg", "text": "cap"}]
        verified = [{"id": "img1.jpg", "text": "vcap"}]
        prepare._verifier.verify.return_value = verified
        loop_mock = MagicMock()
        loop_mock.is_running.return_value = False
        with (
            patch.object(prepare, "_filter_extracted_images", return_value=["img1.jpg"]),
            patch.object(prepare, "_load_captions", return_value=captions),
            patch.object(prepare, "_merge_caption") as mock_merge,
            patch("mx_rag.graphrag.multimodal.extractor.asyncio.get_event_loop", return_value=loop_mock),
        ):
            prepare.extract([self._pdf])
        loop_mock.run_until_complete.assert_called_once()
        prepare._verifier.verify.assert_called_once_with(captions)
        mock_merge.assert_called_once()
        args, _ = mock_merge.call_args
        self.assertEqual(args[1], verified)

    def test_images_captioning_runtime_error_branch(self):
        prepare = self._build()
        prepare._parser.parse.return_value = (["a.md"], ["img1.jpg"])
        with (
            patch.object(prepare, "_filter_extracted_images", return_value=["img1.jpg"]),
            patch.object(prepare, "_load_captions", return_value=[]),
            patch.object(prepare, "_merge_caption"),
            patch("mx_rag.graphrag.multimodal.extractor.asyncio.get_event_loop", side_effect=RuntimeError),
            patch("mx_rag.graphrag.multimodal.extractor.asyncio.run", return_value=[]) as mock_run,
        ):
            prepare.extract([self._pdf])
        mock_run.assert_called_once()

    def test_images_captioning_running_loop_branch(self):
        prepare = self._build()
        prepare._parser.parse.return_value = (["a.md"], ["img1.jpg"])
        loop_mock = MagicMock()
        loop_mock.is_running.return_value = True
        executor_mock = MagicMock()
        executor_mock.return_value.__enter__.return_value.submit.return_value.result.return_value = []
        with (
            patch.object(prepare, "_filter_extracted_images", return_value=["img1.jpg"]),
            patch.object(prepare, "_load_captions", return_value=[]),
            patch.object(prepare, "_merge_caption"),
            patch("mx_rag.graphrag.multimodal.extractor.asyncio.get_event_loop", return_value=loop_mock),
            patch("concurrent.futures.ThreadPoolExecutor", executor_mock),
        ):
            prepare.extract([self._pdf])
        executor_mock.return_value.__enter__.return_value.submit.assert_called_once()

    def test_images_all_already_processed_skips_captioning(self):
        prepare = self._build()
        prepare._parser.parse.return_value = (["a.md"], ["img1.jpg"])
        with (
            patch.object(prepare, "_filter_extracted_images", return_value=[]),
            patch.object(prepare, "_load_captions", return_value=[]),
            patch.object(prepare, "_merge_caption"),
            patch("mx_rag.graphrag.multimodal.extractor.asyncio.get_event_loop") as mock_gel,
        ):
            prepare.extract([self._pdf])
        # no new images -> captioning not invoked, event loop not touched
        prepare._captioner.caption.assert_not_called()
        mock_gel.assert_not_called()


class TestFilterExtractedImages(_BaseExtractorTest):
    def test_no_result_file_returns_all(self):
        prepare = self._build()
        images = ["a.jpg", "b.png"]
        self.assertEqual(prepare._filter_extracted_images(images, "/no/result.json"), images)

    def test_filters_already_extracted(self):
        prepare = self._build()
        result_path = os.path.join(self._output_folder, "tmp.json")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "a.jpg", "text": "done"}) + "\n")
            f.write("\n")  # empty line -> skipped
            f.write("not-json\n")  # invalid json -> skipped
            f.write(json.dumps({"id": "b.png", "text": ""}) + "\n")  # no text -> not counted
        remaining = prepare._filter_extracted_images(["a.jpg", "b.png", "c.png"], result_path)
        self.assertEqual(remaining, ["b.png", "c.png"])

    def test_read_exception_returns_all(self):
        prepare = self._build()
        result_path = os.path.join(self._output_folder, "tmp.json")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write("{}")
        images = ["a.jpg", "b.png"]
        with patch("builtins.open", side_effect=OSError("boom")):
            self.assertEqual(prepare._filter_extracted_images(images, result_path), images)


class TestMergeCaption(_BaseExtractorTest):
    def _setup_md(self, content="![旧描述](images/img.png)"):
        sub = os.path.join(self._output_folder, "doc1", "sub")
        os.makedirs(sub, exist_ok=True)
        md_path = os.path.join(sub, "note.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)
        image_id = os.path.join(self._output_folder, "doc1", "sub", "images", "img.png")
        return md_path, image_id

    def test_merge_replace_md(self):
        md_path, image_id = self._setup_md()
        prepare = self._build()
        captions = [{"id": image_id, "text": "新描述"}]
        prepare._merge_caption(self._output_folder, captions, "replace")
        with open(os.path.join(self._output_folder, "note.md"), "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "新描述")

    def test_merge_replace_md_without_alt(self):
        md_path, image_id = self._setup_md("![](images/img.png)")
        prepare = self._build()
        captions = [{"id": image_id, "text": "新描述"}]
        prepare._merge_caption(self._output_folder, captions, "replace")
        with open(os.path.join(self._output_folder, "note.md"), "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "新描述")

    def test_merge_replace_plain_text_unchanged(self):
        # 非 Markdown 图片语法的普通文本引用不被替换
        md_path, image_id = self._setup_md("ref images/img.png here")
        prepare = self._build()
        captions = [{"id": image_id, "text": "新描述"}]
        prepare._merge_caption(self._output_folder, captions, "replace")
        with open(os.path.join(self._output_folder, "note.md"), "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "ref images/img.png here")

    def test_merge_replace_partial_match(self):
        # 图片文件名是 img.png, img_big.png 不应被误匹配
        md_path, image_id = self._setup_md("![x](images/img.png) and ![y](images/img_big.png)")
        prepare = self._build()
        captions = [{"id": image_id, "text": "仅替换img"}]
        prepare._merge_caption(self._output_folder, captions, "replace")
        with open(os.path.join(self._output_folder, "note.md"), "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "仅替换img and ![y](images/img_big.png)")

    def test_merge_append_md(self):
        md_path, image_id = self._setup_md()
        prepare = self._build()
        captions = [{"id": image_id, "text": "CAP"}]
        prepare._merge_caption(self._output_folder, captions, "append")
        with open(os.path.join(self._output_folder, "note.md"), "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("![旧描述](images/img.png)", content)
        self.assertIn("# CAP", content)

    def test_merge_no_match_leaves_content_unchanged(self):
        md_path, _ = self._setup_md()
        prepare = self._build()
        # different doc folder -> parts[-4] != md_path.parts[-3]
        captions = [{"id": os.path.join(self._output_folder, "other", "sub", "images", "x.png"), "text": "X"}]
        prepare._merge_caption(self._output_folder, captions, "replace")
        with open(os.path.join(self._output_folder, "note.md"), "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "![旧描述](images/img.png)")

    def test_merge_append_txt(self):
        sub = os.path.join(self._output_folder, "doc1", "sub")
        os.makedirs(sub, exist_ok=True)
        txt_path = os.path.join(sub, "notes.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("base")
        image_id = os.path.join(self._output_folder, "doc1", "sub", "images", "img.png")
        prepare = self._build()
        prepare._merge_caption(self._output_folder, [{"id": image_id, "text": "CAP"}], "append")
        with open(os.path.join(self._output_folder, "notes.txt"), "r", encoding="utf-8") as f:
            content = f.read()
        self.assertTrue(content.startswith("base"))
        self.assertIn("CAP", content)

    def test_merge_replace_txt_skipped(self):
        sub = os.path.join(self._output_folder, "doc1", "sub")
        os.makedirs(sub, exist_ok=True)
        txt_path = os.path.join(sub, "notes.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("base")
        image_id = os.path.join(self._output_folder, "doc1", "sub", "images", "img.png")
        prepare = self._build()
        prepare._merge_caption(self._output_folder, [{"id": image_id, "text": "CAP"}], "replace")
        # replace merge_type on txt files is skipped; no merged file written at data_set root
        self.assertFalse(os.path.isfile(os.path.join(self._output_folder, "notes.txt")))


class TestLoadCaptions(_BaseExtractorTest):
    def test_file_not_exist_returns_empty(self):
        self.assertEqual(MultimodalPrepare._load_captions("/no/such.json"), [])

    def test_parses_valid_lines(self):
        path = os.path.join(self._output_folder, "c.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "a", "text": "t1"}) + "\n")
            f.write("\n")
            f.write("bad\n")
            f.write(json.dumps({"id": "b", "text": ""}) + "\n")
        self.assertEqual(len(MultimodalPrepare._load_captions(path)), 1)


if __name__ == "__main__":
    unittest.main()
