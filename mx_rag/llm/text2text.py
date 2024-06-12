# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import json
from urllib.parse import urljoin

from loguru import logger

from mx_rag.utils import is_english, RequestUtils, safe_get


class Text2TextLLM:
    HEADER = {
        "Content-Type": "application/json"
    }

    def __init__(self, url: str, model_name: str, timeout: int = 10, max_prompt_len=128 * 1024 * 1024):
        self._model_name = model_name
        self._url = url
        self._client = RequestUtils(timeout=timeout)
        self._max_prompt_len = max_prompt_len

    def get_request_body(self, query: str, history: list[dict], role: str = "user", **kwargs):
        history.append({"role": role, "content": query})

        request_body = {
            "model": self._model_name,
            "messages": history,
            "max_tokens": kwargs.get("max_tokens", 16),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "seed": kwargs.get("seed", None),
            "temperature": kwargs.get("temperature", 1),
            "top_p": kwargs.get("top_p", 1),
        }
        return request_body

    def chat(self, query: str, history: list[dict] = None, role: str = "role", **kwargs):
        ans = ""
        if query is None:
            logger.error(f"query cannot be None")
            return ans

        if len(query) > self._max_prompt_len or len(query) == 0:
            logger.error(f"query content len [{len(query)}] not in (0, {self._max_prompt_len}]")
            return ans

        request_body = self.get_request_body(query, history, role, **kwargs)
        request_body["stream"] = False
        chat_url = urljoin(self._url, "v1/chat/completions")
        response = self._client.post(url=chat_url, body=json.dumps(request_body), headers=self.HEADER)
        if response.success:
            try:
                data = json.loads(response.data)
            except json.JSONDecodeError as e:
                logger.error(f"response content cannot convert to json format: {e}")
                return ans
            ans = safe_get(data, ["choices", 0, "message", "content"], "")
            if safe_get(data, ["choices", 0, "finish_reason"], "") == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                            [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
        else:
            logger.error("get response failed")
        return ans

    def chat_streamly(self, query: str, history: list[dict], role: str = "role", **kwargs):
        ans = ""
        if query is None or len(query) > self._max_prompt_len:
            logger.error(f"query cannot be None or content len not in  (0, {self._max_prompt_len})")
            yield ans
            return
        request_body = self.get_request_body(query, history, role, **kwargs)
        request_body["stream"] = True
        chat_url = urljoin(self._url, "v1/chat/completions")
        ans = ""
        response = self._client.post_streamly(url=chat_url, body=json.dumps(request_body), headers=self.HEADER)
        for result in response:
            if not result.success:
                logger.error("get response failed")
                break
            chunk = result.data
            if not chunk.strip() or not chunk.startswith(b"data:"):
                continue
            try:
                data = json.loads(chunk[6:].decode("utf-8").strip())
            except json.JSONDecodeError as e:
                logger.error(f"response content cannot convert to json format:{e}")
                break
            finish_reason = safe_get(data, ["choices", 0, "finish_reason"], "")
            if finish_reason == "stop":
                break
            elif finish_reason == "length":
                ans += "...\nFor the content length reason, it stopped, continue?" if is_english(
                    [ans]) else "······\n由于长度的原因，回答被截断了，要继续吗？"
                yield ans
                break
            elif finish_reason == "":
                break
            ans += safe_get(data, ["choices", 0, "delta", "content"], "")
            yield ans
