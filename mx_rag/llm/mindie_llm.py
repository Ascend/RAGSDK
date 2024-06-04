# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
from urllib.parse import urljoin

from loguru import logger

from mx_rag.utils import is_english, RequestUtils, safe_get


class MindieLLM:
    HEADER = {
        "Content-Type": "application/json"
    }
    MAX_QUERY_LEN = 16000

    def __init__(self, model_name: str, url: str, timeout: int = 10):
        self.model_name = model_name
        self.url = url
        self.client = RequestUtils(timeout=timeout)

    def get_request_body(self, query: str, history: list[dict], role: str = "role", **kwargs):
        history.insert(0, {"role": role, "content": query})

        request_body = {
            "model": self.model_name,
            "messages": history,
            "max_tokens": kwargs.get("max_tokens", 16),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "seed": kwargs.get("seed", None),
            "temperature": kwargs.get("temperature", 1),
            "top_p": kwargs.get("top_p", 1),
        }
        return request_body

    def chat(self, query: str, history: list[dict], role: str = "role", **kwargs):
        ans = ""
        if query is None or len(query) > MindieLLM.MAX_QUERY_LEN:
            logger.error(f"query cannot be None or content len not in (0, {MindieLLM.MAX_QUERY_LEN}]")
            return ans
        request_body = self.get_request_body(query, history, role, **kwargs)
        request_body["stream"] = False
        chat_url = urljoin(self.url, "v1/chat/completions")
        response = self.client.post(url=chat_url, body=json.dumps(request_body), headers=MindieLLM.HEADER)
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
        if query is None or len(query) > MindieLLM.MAX_QUERY_LEN:
            logger.error(f"query cannot be None or content len not in (0, {MindieLLM.MAX_QUERY_LEN}]")
            yield ans
            return
        request_body = self.get_request_body(query, history, role, **kwargs)
        request_body["stream"] = True
        chat_url = urljoin(self.url, "v1/chat/completions")
        ans = ""
        response = self.client.post_streamly(url=chat_url, body=json.dumps(request_body), headers=MindieLLM.HEADER)
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
