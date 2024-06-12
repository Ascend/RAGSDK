# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import Union, Iterator, List, Dict

from loguru import logger

from mx_rag.chain import SimpleRetrieval


class MultiTurnRetrieval(SimpleRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_history = 20
        self._role = "user"
        self._history: List[Dict] = []

    def query(self,
              text: str,
              max_tokens: int,
              temperature: float,
              top_p: float,
              stream: bool = False) -> Union[str, Iterator]:
        if len(self._history) >= self._max_history:
            self._history.pop(0)

        if self._content != "":
            self._history.append({"role": "assistant", "content": self._content})
            self._content = ""

        return self._query(text, max_tokens, temperature, top_p, stream)
