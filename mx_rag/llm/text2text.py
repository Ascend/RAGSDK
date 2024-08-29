# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import json
import sys
from typing import List, Optional, Any, Iterator

from pydantic.v1 import Field

from langchain.llms.base import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from loguru import logger

from mx_rag.utils.common import safe_get, MB, INT_32_MAX, validate_params, MAX_URL_LENGTH, MAX_MODEL_NAME_LENGTH
from mx_rag.utils.url import RequestUtils

HEADER = {
    "Content-Type": "application/json"
}


def _check_sys_messages(sys_messages: List[dict] = None) -> bool:
    if sys_messages is None:
        return True

    if len(sys_messages) > 16:
        return False

    for d in sys_messages:
        if not isinstance(d, dict) or len(d) > 16:
            return False
        for k, v in d.items():
            if len(k) > 16 or len(v) > 4 * MB:
                return False
    return True


class Text2TextLLM(LLM):
    base_url: str = Field(min_length=1, max_length=MAX_URL_LENGTH)
    model_name: str = Field(min_length=1, max_length=MAX_MODEL_NAME_LENGTH)
    timeout: int = Field(ge=1, le=INT_32_MAX, default=10)
    cert_file: str = Field(min_length=0, max_length=128, default="")
    crl_file: str = Field(min_length=0, max_length=128, default="")
    use_http: bool = Field(default=False)
    max_tokens: int = 512
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0

    @property
    def _client(self):
        return RequestUtils(timeout=self.timeout, cert_file=self.cert_file, crl_file=self.crl_file,
                            use_http=self.use_http)

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

    @validate_params(
        query=dict(validator=lambda x: 0 < len(x) <= 4 * MB),
        sys_messages=dict(validator=lambda x: _check_sys_messages(x)),
        role=dict(validator=lambda x: 0 < len(x) <= 16),
    )
    def chat(self, query: str,
             sys_messages: List[dict] = None,
             role: str = "user",
             **kwargs):
        ans = ""
        if sys_messages is None:
            sys_messages = []
        request_body = self._get_request_body(query, sys_messages, role, **kwargs)
        request_body["stream"] = False
        response = self._client.post(url=self.base_url, body=json.dumps(request_body), headers=HEADER)
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

    @validate_params(
        query=dict(validator=lambda x: 0 < len(x) <= 4 * MB),
        sys_messages=dict(validator=lambda x: _check_sys_messages(x)),
        role=dict(validator=lambda x: 0 < len(x) <= 16),
    )
    def chat_streamly(self, query: str,
                      sys_messages: List[dict] = None,
                      role: str = "user",
                      **kwargs):
        if sys_messages is None:
            sys_messages = []

        request_body = self._get_request_body(query, sys_messages, role, **kwargs)
        request_body["stream"] = True
        ans = ""
        response = self._client.post_streamly(url=self.base_url, body=json.dumps(request_body), headers=HEADER)
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

    def _get_request_body(self, query: str, messages: List[dict], role: str, **kwargs):
        messages.append({"role": role, "content": query})

        max_tokens = "max_tokens"
        presence_penalty = "presence_penalty"
        frequency_penalty = "frequency_penalty"
        temperature = "temperature"
        top_p = "top_p"

        seed_str = "seed"

        seed = kwargs.get(seed_str, None)
        if seed is not None:
            seed = self._validate_range(kwargs.get(seed_str, None), (0, INT_32_MAX), int,
                                        inclusive_min=False, param_name=seed_str)
        # 适配MindIE参数范围
        request_body = {
            "model": self.model_name,
            "messages": messages,
            max_tokens: self._validate_range(kwargs.get(max_tokens, 512), (1, INT_32_MAX), int,
                                             inclusive_min=False, param_name=max_tokens),
            presence_penalty: self._validate_range(kwargs.get(presence_penalty, 0.0), (-2.0, 2.0), float,
                                                   param_name=presence_penalty),
            frequency_penalty: self._validate_range(kwargs.get(frequency_penalty, 0.0), (-2.0, 2.0), float,
                                                    param_name=frequency_penalty),
            seed_str: seed,
            temperature: self._validate_range(kwargs.get(temperature, 1.0), (0.0, 2.0), float,
                                              inclusive_min=False, param_name=temperature),
            top_p: self._validate_range(kwargs.get(top_p, 1.0), (0.0, 1.0), float,
                                        inclusive_min=False, param_name=top_p)
        }
        return request_body

    def _llm_type(self):
        return self.model_name

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            callbacks: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        return self.chat(prompt, max_tokens=self.max_tokens, temperature=self.temperature,
                         presence_penalty=self.presence_penalty, frequency_penalty=self.frequency_penalty,
                         top_p=self.top_p)

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for response in self.chat_streamly(prompt, max_tokens=self.max_tokens, temperature=self.temperature,
                                           presence_penalty=self.presence_penalty,
                                           frequency_penalty=self.frequency_penalty,
                                           top_p=self.top_p):
            yield GenerationChunk(text=response)
