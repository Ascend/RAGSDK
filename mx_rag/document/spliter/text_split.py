# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import re
from typing import List, Iterable, Callable


class TextSplitterBase:
    default_chunk_size = 4000
    default_chunk_overlap = 200

    def __init__(self,
                 chunk_size: int = default_chunk_size,
                 chunk_overlap: int = default_chunk_overlap,
                 len_func: Callable[[str], int] = len,
                 keep_separator: bool = False,
                 separator_after_split_text: bool = False,
                 strip_whitespace: bool = True):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # 计算列表个数的回调函数
        self._len_func = len_func

        # 是否保留分隔符在分割字符串中
        self._keep_separator = keep_separator

        # 分隔符位于分割字符串的后面还是前面，对于中文而言 分隔符位于后面更加合适
        self._separator_after_split_text = separator_after_split_text

        self._strip_whitespace = strip_whitespace

    @staticmethod
    def _split_text_and_no_keep_seperator(text, separator) -> List[str]:
        res_split_text = re.split(separator, text)
        return [s for s in res_split_text if s]

    def _split_text_and_keep_seperator(self, text, separator) -> List[str]:
        split_text = re.split(f"({separator})", text)
        if not self._separator_after_split_text:
            res_split_text = [split_text[i] + split_text[i + 1] for i in range(1, len(split_text), 2)]
            res_split_text = (res_split_text + split_text[-1:]) if len(split_text) % 2 == 0 else res_split_text
            res_split_text = [split_text[0]] + res_split_text
        else:
            res_split_text = [split_text[i] + split_text[i + 1] for i in range(0, len(split_text) - 1, 2)]
            res_split_text = res_split_text if len(split_text) % 2 == 0 else (res_split_text + split_text[-1:])

        return [s for s in res_split_text if s]

    def _split_text_with_reg(self, text: str, separator: str) -> List[str]:
        if separator and self._keep_separator:
            return self._split_text_and_keep_seperator(text, separator)

        if separator:
            return self._split_text_and_no_keep_seperator(text, separator)

        return [s for s in list(text) if s]

    def _split_text(self, text, separator: str) -> List[str]:
        return self._split_text_with_reg(text, separator)

    def _join_split_to_chunk(self, chunk_list, chunk, separator):
        chunk_text = separator.join(chunk)
        if self._strip_whitespace:
            chunk_text = chunk_text.strip()
        if chunk_text != "":
            chunk_list.append(chunk_text)

    def _merge_split_overlap_to_chunk(self, chunk, chunk_used_size, text_split_size, separator_len):
        while chunk_used_size > self._chunk_overlap or (
                chunk_used_size + text_split_size + (separator_len if len(chunk) > 0 else 0)
                > self._chunk_size) and chunk_used_size > 0:
            chunk_used_size -= self._len_func(chunk[0]) + (
                separator_len if len(chunk) > 1 else 0
            )
            chunk = chunk[1:]

        return chunk_used_size, chunk

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        chunk_list = []
        chunk: List[str] = []

        chunk_used_size = 0
        separator = "" if self._keep_separator else separator
        separator_len = self._len_func(separator)

        for split_text in splits:
            text_split_size = self._len_func(split_text)
            if (chunk_used_size + text_split_size + (separator_len if len(chunk) > 0 else 0)
                    > self._chunk_size):
                if len(chunk) > 0:
                    # chunk 大于了 chunk_size 则先把已经merge的split 合并到一个chunk
                    self._join_split_to_chunk(chunk_list, chunk, separator)

                    # 然后根据overlap选择保留一部分和新的split组合成新的chunk
                    chunk_used_size, chunk = self._merge_split_overlap_to_chunk(chunk,
                                                                                chunk_used_size,
                                                                                text_split_size,
                                                                                separator_len)
            chunk_used_size += text_split_size + (separator_len if len(chunk) > 0 else 0)
            chunk.append(split_text)

        self._join_split_to_chunk(chunk_list, chunk, separator)
        return chunk_list
