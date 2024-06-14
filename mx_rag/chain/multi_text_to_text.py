# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import Union, Iterator, List, Dict

from loguru import logger

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
        self._iter = 1

    def query(self, text: str, *args, **kwargs) -> Union[Dict, Iterator[Dict]]:
        logger.info(f"current iter is {self._iter}")
        if self._iter >= self._max_history:
            # remove role user
            self._history.pop(0)
            if self._content != "":
                # remove role assistant
                self._history.pop(0)
            self._iter -= 1

        # previous reqeust was successful, append assistant
        if self._content != "":
            self._history.append({"role": "assistant", "content": self._content})
            self._content = ""
            self._iter += 1
        elif self._iter != 1 and self._iter != self._max_history - 1:  # previous reqeust was failed, remove role user
            self._history.pop(0)

        return self._query(text, *args, **kwargs)
