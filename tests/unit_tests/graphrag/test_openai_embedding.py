#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for mx_rag.graphrag.multimodal.openai_embedding.OpenAIEmbedding cases."""

import base64
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from paddle.base import libpaddle  # noqa: F401
from mx_rag.graphrag.multimodal.openai_embedding import OpenAIEmbedding


class TestOpenAIEmbeddingClient(unittest.TestCase):
    def test_init_with_url_and_api_key(self):
        emb = OpenAIEmbedding(model_name="model", url="http://srv/v1", api_key="test-key")
        self.assertEqual(emb.url, "http://srv/v1")
        self.assertEqual(emb.model_name, "model")
        self.assertIsNone(emb._client)

    def test_init_with_client(self):
        mock_client = MagicMock()
        mock_client.base_url = "http://custom/v1"
        emb = OpenAIEmbedding(model_name="model", client=mock_client)
        self.assertIs(emb._client, mock_client)
        self.assertEqual(emb.url, "http://custom/v1")

    def test_init_missing_both_raises_error(self):
        with self.assertRaises(ValueError):
            OpenAIEmbedding(model_name="model")

    def test_get_client_lazy_and_cached(self):
        emb = OpenAIEmbedding(model_name="model", url="http://srv/v1", api_key="test-key")
        with patch("mx_rag.graphrag.multimodal.openai_embedding.OpenAI") as mock_openai:
            client1 = emb._get_client()
            client2 = emb._get_client()
            self.assertIs(client1, client2)
            mock_openai.assert_called_once_with(api_key="test-key", base_url="http://srv/v1")


class TestCreateChatEmbeddings(unittest.TestCase):
    def test_create_chat_embeddings_builds_body(self):
        client = MagicMock()
        messages = [{"role": "user", "content": "hi"}]
        result = OpenAIEmbedding.create_chat_embeddings(
            client,
            messages=messages,
            model="m",
            encoding_format="float",
            continue_final_message=True,
            add_special_tokens=True,
        )
        self.assertIs(result, client.post.return_value)
        client.post.assert_called_once()
        _, kwargs = client.post.call_args
        self.assertEqual(kwargs["cast_to"].__name__, "CreateEmbeddingResponse")
        self.assertEqual(kwargs["body"]["model"], "m")
        self.assertEqual(kwargs["body"]["messages"], messages)
        self.assertTrue(kwargs["body"]["continue_final_message"])
        self.assertTrue(kwargs["body"]["add_special_tokens"])


class TestEncodeImageToBase64(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self._tmp.cleanup)

    def _write(self, name, data=b"fake"):
        path = os.path.join(self._tmp.name, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            OpenAIEmbedding.encode_image_to_base64("/no/such/file.jpg")

    def test_unsupported_format_raises(self):
        path = self._write("img.tiff")
        with self.assertRaises(ValueError):
            OpenAIEmbedding.encode_image_to_base64(path)

    def test_supported_jpg(self):
        path = self._write("img.jpg", b"\x89PNGdata")
        encoded = OpenAIEmbedding.encode_image_to_base64(path)
        self.assertTrue(encoded.startswith("data:image/jpg;base64,"))
        payload = encoded.split(";base64,", 1)[1]
        self.assertEqual(base64.b64decode(payload), b"\x89PNGdata")

    def test_no_extension_defaults_png(self):
        path = self._write("noext", b"abc")
        encoded = OpenAIEmbedding.encode_image_to_base64(path)
        self.assertTrue(encoded.startswith("data:image/png;base64,"))


class TestBuildMessages(unittest.TestCase):
    def test_build_text_messages_structure(self):
        msgs = OpenAIEmbedding._build_text_messages("hello")
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"][0]["type"], "text")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[1]["content"][0]["text"], "hello")
        self.assertEqual(msgs[2]["role"], "assistant")

    def test_build_image_messages_structure(self):
        with patch.object(
            OpenAIEmbedding,
            "encode_image_to_base64",
            return_value="data:image/png;base64,AAA",
        ):
            msgs = OpenAIEmbedding._build_image_messages("/tmp/img.png")
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[1]["content"][0]["type"], "image_url")
        self.assertEqual(msgs[1]["content"][0]["image_url"]["url"], "data:image/png;base64,AAA")
        self.assertEqual(msgs[1]["content"][1]["type"], "text")


class TestEmbedConcurrent(unittest.TestCase):
    def _make_embed(self):
        emb = OpenAIEmbedding(model_name="model", url="http://srv/v1", api_key="test-key")
        emb._client = MagicMock()
        return emb

    def test_empty_items_returns_empty(self):
        emb = self._make_embed()
        self.assertEqual(emb._embed_concurrent([], OpenAIEmbedding._build_text_messages), [])

    def test_run_text_preserves_order(self):
        emb = self._make_embed()

        def fake_create(client, *, messages, model, **kwargs):
            resp = MagicMock()
            text = messages[1]["content"][0]["text"]
            resp.data = [MagicMock(embedding=[float(len(text))])]
            return resp

        with patch.object(OpenAIEmbedding, "create_chat_embeddings", side_effect=fake_create):
            result = emb.run_text(["a", "bb", "ccc"])
        self.assertEqual(result, [[1.0], [2.0], [3.0]])

    def test_run_image_returns_embeddings(self):
        emb = self._make_embed()
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[0.5, 0.5])]
        with (
            patch.object(OpenAIEmbedding, "create_chat_embeddings", return_value=resp),
            patch.object(
                OpenAIEmbedding,
                "encode_image_to_base64",
                return_value="data:image/png;base64,AAA",
            ),
        ):
            result = emb.run_image(["/tmp/a.png", "/tmp/b.png"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.5, 0.5])

    def test_embed_concurrent_filters_none_results(self):
        emb = self._make_embed()
        resp_ok = MagicMock()
        resp_ok.data = [MagicMock(embedding=[1.0])]
        resp_none = MagicMock()
        resp_none.data = [MagicMock(embedding=None)]

        def fake_create(client, *, messages, model, **kwargs):
            text = messages[1]["content"][0]["text"]
            return resp_ok if text == "keep" else resp_none

        with patch.object(OpenAIEmbedding, "create_chat_embeddings", side_effect=fake_create):
            result = emb.run_text(["keep", "drop"])
        # _embed returned None for "drop"; None entries are filtered out.
        self.assertEqual(result, [[1.0]])


if __name__ == "__main__":
    unittest.main()
