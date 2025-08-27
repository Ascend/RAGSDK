# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import json
from typing import List, Optional, Any
from pydantic import ConfigDict, Field

from langchain.llms.base import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from loguru import logger

from mx_rag.llm.text2text import _check_sys_messages
from mx_rag.utils import ClientParam
from mx_rag.utils.common import safe_get, MB, validate_params, MAX_URL_LENGTH, MAX_MODEL_NAME_LENGTH, MAX_PROMPT_LENGTH
from mx_rag.llm.llm_parameter import LLMParameterConfig
from mx_rag.utils.url import RequestUtils

IMG_TO_TEXT_PROMPT = '''Given an image containing a table or figure, please provide a structured and detailed
description in chinese with two levels of granularity:

  Coarse-grained Description:
  - Summarize the overall content and purpose of the image.
  - Briefly state what type of data or information is presented (e.g., comparison, trend, distribution).
  - Mention the main topic or message conveyed by the table or figure.

  Fine-grained Description:
  - Describe the specific details present in the image.
  - For tables: List the column and row headers, units, and any notable values, patterns, or anomalies.
  - For figures (e.g., plots, charts): Explain the axes, data series, legends, and any significant trends, outliers,
  or data points.
  - Note any labels, captions, or annotations included in the image.
  - Highlight specific examples or noteworthy details.

  Deliver the description in a clear, organized, and reader-friendly manner, using bullet points or paragraphs
  as appropriate, answer in chinese'''


def _check_image_url(image_url):
    # 检查输入是否为字典
    if not isinstance(image_url, dict):
        return False

    # 检查是否包含 url 键
    if "url" not in image_url or len(image_url) != 1:
        return False

    # 检查 url 的值是否为字符串
    url_value = image_url["url"]
    if not isinstance(url_value, str):
        return False

    # 检查字符串长度
    if not 0 < len(url_value) <= 4*MB:
        return False

    return True


class Img2TextLLM(LLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    base_url: str = Field(min_length=1, max_length=MAX_URL_LENGTH)
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_LENGTH, default=IMG_TO_TEXT_PROMPT)
    model_name: str = Field(min_length=1, max_length=MAX_MODEL_NAME_LENGTH)
    llm_config: LLMParameterConfig = LLMParameterConfig()
    client_param: ClientParam = ClientParam()

    @property
    def _client(self):
        return RequestUtils(client_param=self.client_param)

    @validate_params(
        image_url=dict(validator=lambda x: _check_image_url(x),
                       message="param must be dict, and len(dict)==1"
                               "and must contain 'url' key with string value less than 4MB"),
        sys_messages=dict(validator=lambda x: _check_sys_messages(x),
                          message="param must be None or List[dict], and length of dict <= 16, "
                                  "k-v of dict: len(k) <=16 and len(v) <= 4 * MB"),
        role=dict(validator=lambda x: 1 <= len(x) <= 16, message="param length range [1, 16]"),
        llm_config=dict(validator=lambda x: isinstance(x, LLMParameterConfig),
                        message="param must be LLMParameterConfig")
    )
    def chat(self, image_url: dict,
             sys_messages: List[dict] = None,
             role: str = "user",
             llm_config: LLMParameterConfig = LLMParameterConfig()):
        ans = ""
        if sys_messages is None:
            sys_messages = []
        request_body = self._get_request_body(self.prompt, image_url, sys_messages, role, llm_config)
        request_body["stream"] = False
        response = self._client.post(url=self.base_url, body=json.dumps(request_body),
                                     headers={"Content-Type": "application/json"})
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

    def _get_request_body(self, prompt: str, image_url: dict, messages: List[dict], role: str,
                          llm_config: LLMParameterConfig):
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": image_url}
        ]
        messages.append({"role": role, "content": content})
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": llm_config.max_tokens,
            "presence_penalty": llm_config.presence_penalty,
            "frequency_penalty": llm_config.frequency_penalty,
            "seed": llm_config.seed,
            "temperature": llm_config.temperature,
            "top_p": llm_config.top_p
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
        return self.chat(prompt, llm_config=self.llm_config)
