# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import re
from typing import Any, List

from mx_rag.document.splitter.text_splitter import TextSplitterBase


class CharTextSplitter(TextSplitterBase):
    def __init__(
            self, separator: str = "\n\n", is_separator_regex: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._separator = separator
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        separator = (
            self._separator if self._is_separator_regex else re.escape(self._separator)
        )
        splits = self._split_text(text, separator)
        return self._merge_splits(splits, self._separator)
