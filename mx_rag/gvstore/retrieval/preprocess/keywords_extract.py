# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from loguru import logger

from mx_rag.gvstore.prompt.prompt_template import PROMPTS
from mx_rag.llm import Text2TextLLM
from mx_rag.utils.common import validate_params


class KeywordsExtract:
    @validate_params(
        llm=dict(validator=lambda x: isinstance(x, Text2TextLLM), message="param must be instance of Text2TextLLM")
    )
    def __init__(self, llm: Text2TextLLM, **kwargs) -> None:
        self.llm = llm
        self.keyword_extract_prompt = PROMPTS["KEYWORDS_EXTRACT"]

    def extract_keywords(self, question, **kwargs):
        try:
            prompt = self.keyword_extract_prompt.format(text=question)
            result = self.llm.chat(prompt, llm_config=self.llm.llm_config)
            result = result.split(';')[0].split(',')
            result = [r.strip() for r in result]
        except Exception as e:
            logger.error(f"extract keywords got exception: {e}")
            result = [question]
        return result
