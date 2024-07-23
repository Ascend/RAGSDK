# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import json
import sys
from typing import List

from loguru import logger

from mx_rag.utils.common import safe_get
from mx_rag.utils.url import RequestUtils


class Text2TextLLM:
    HEADER = {
        "Content-Type": "application/json"
    }
    INT64_MAX = (1 << 63) - 1

    def __init__(self,
                 url: str,
                 model_name: str,
                 timeout: int = 10,
                 max_prompt_len=128 * 1024 * 1024,
                 max_history_len=100,
                 cert_file: str = "",
                 crl_file: str = "",
                 use_http: bool = False):
        self._model_name = model_name
        self._url = url
        self._client = RequestUtils(timeout=timeout, cert_file=cert_file, crl_file=crl_file, use_http=use_http)
        self._max_history_len = max_history_len
        self._max_prompt_len = max_prompt_len

    @staticmethod
    def _validate_range(value, value_range, expected_type, inclusive_min=True, param_name=""):
        if value is None:
            raise ValueError(f"{param_name} cannot be None.")

        if not isinstance(value, expected_type):
            raise TypeError(f"{param_name}={value} is of incorrect type (expected {expected_type.__name__}).")

        min_value, max_value = value_range

        if inclusive_min:
            if value < min_value:
                raise ValueError(
                    f"{param_name}={value} is less than minimum allowed value ({min_value}).")
        else:
            if value <= min_value:
                raise ValueError(
                    f"{param_name}={value} is less than or equal to the minimum allowed value ({min_value}).")

        if value > max_value:
            raise ValueError(
                f"{param_name}={value} is greater than the maximum allowed value ({max_value}).")

        return value

    @staticmethod
    def _validate_history_format(history: List[dict]):
        if history is None:
            return False

        required_keys = {"role", "content"}

        for item in history:
            if not isinstance(item, dict):
                return False
            if set(item.keys()) != required_keys:
                return False

        return True

    def chat(self, query: str, history: List[dict] = None, role: str = "user", **kwargs):
        ans = ""
        if query is None:
            logger.error(f"query cannot be None")
            return ans

        if len(query) > self._max_prompt_len or len(query) == 0:
            logger.error(f"query content len [{len(query)}] not in (0, {self._max_prompt_len}]")
            return ans
        if history is None:
            history = []
        request_body = self._get_request_body(query, history, role, **kwargs)
        request_body["stream"] = False
        response = self._client.post(url=self._url, body=json.dumps(request_body), headers=self.HEADER)
        if response.success:
            try:
                data = json.loads(response.data)
            except json.JSONDecodeError as e:
                logger.error(f"response content cannot convert to json format: {e}")
                return ans
            except Exception as e:
                logger.error(f"json load error: {e}")
                return ans

            ans = safe_get(data, ["choices", 0, "message", "content"], "")
            if safe_get(data, ["choices", 0, "finish_reason"], "") == "length":
                logger.info("for the content length reason, it stopped.")
                ans += "......"
        else:
            logger.error("get response failed, please check the server log for details")
        return ans

    def chat_streamly(self, query: str, history: List[dict] = None, role: str = "user", **kwargs):
        ans = ""
        if query is None or len(query) > self._max_prompt_len:
            logger.error(f"query cannot be None or content len not in  (0, {self._max_prompt_len})")
            yield ans
            return
        if history is None:
            history = []

        request_body = self._get_request_body(query, history, role, **kwargs)
        request_body["stream"] = True
        ans = ""
        response = self._client.post_streamly(url=self._url, body=json.dumps(request_body), headers=self.HEADER)
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
            except Exception as e:
                logger.error(f"json load error: {e}")
                break

            finish_reason = safe_get(data, ["choices", 0, "finish_reason"], "")
            if finish_reason == "stop":
                break
            elif finish_reason == "length":
                logger.info("for the content length reason, it stopped.")
                ans += "......"
                yield ans
                break
            elif finish_reason == "":
                break
            ans += safe_get(data, ["choices", 0, "delta", "content"], "")
            yield ans

    def _get_request_body(self, query: str, history: List[dict], role: str, **kwargs):
        if len(history) > self._max_history_len:
            raise ValueError(f"The length of the history parameter cannot exceed {self._max_history_len}")

        if not self._validate_history_format(history):
            raise ValueError("the history parameter is not valid, can only contain role and context")

        history.append({"role": role, "content": query})

        max_tokens = "max_tokens"
        presence_penalty = "presence_penalty"
        frequency_penalty = "frequency_penalty"
        temperature = "temperature"
        top_p = "top_p"

        seed_str = "seed"

        seed = kwargs.get(seed_str, None)
        if seed is not None:
            seed = self._validate_range(kwargs.get(seed_str, None), (0, self.INT64_MAX), int,
                                        inclusive_min=False, param_name=seed_str)
        # 适配MindIE参数范围
        request_body = {
            "model": self._model_name,
            "messages": history,
            max_tokens: self._validate_range(kwargs.get(max_tokens, 16), (1, self.INT64_MAX), int,
                                             inclusive_min=False, param_name=max_tokens),
            presence_penalty: self._validate_range(kwargs.get(presence_penalty, 0.0), (-2.0, 2.0), float,
                                                   param_name=presence_penalty),
            frequency_penalty: self._validate_range(kwargs.get(frequency_penalty, 0.0), (-2.0, 2.0), float,
                                                    param_name=frequency_penalty),
            seed_str: seed,
            temperature: self._validate_range(kwargs.get(temperature, 1.0), (0.0, sys.float_info.max), float,
                                              inclusive_min=False, param_name=temperature),
            top_p: self._validate_range(kwargs.get(top_p, 1.0), (0.0, 1.0), float,
                                        inclusive_min=False, param_name=top_p)
        }
        return request_body
