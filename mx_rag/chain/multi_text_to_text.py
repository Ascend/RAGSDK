# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import Union, Iterator, List, Dict

from mx_rag.chain import Chain
from mx_rag.chain.single_text_to_text import SingleText2TextChain


class MultiText2TextChain(SingleText2TextChain, Chain):
    """ 查询带历史记录，大模型输出文本 """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_history = 20
        self._role = "user"
        self._history: List[Dict] = []
        self._content = ""

    def query(self, text : str, *args, **kwargs) -> Union[str, Iterator]:
        if len(self._history) >= self._max_history:
            self._history.pop(0)

        if self._content != "":
            self._history.append({"role": "assistant", "content": self._content})
            self._content = ""

        return self._query(text, *args, **kwargs)
