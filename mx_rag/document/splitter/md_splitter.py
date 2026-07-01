#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

RAGSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from mx_rag.utils.common import validate_params, HEADER_MARK, MAX_SPLIT_SIZE


class ProtectedBlockExtractor:
    """提取受保护的块（表格、代码片段），不使用正则表达式"""

    def __init__(self, text: str):
        self.text = text
        self.length = len(text)
        self.pos = 0

    def extract_all(self):
        """提取所有受保护的块"""
        blocks = []  # [(start, end, block_type), ...]

        while self.pos < self.length:
            # 1. 尝试 HTML 表格
            if self._try_html_table(blocks):
                continue

            # 2. 尝试 Markdown 表格
            if self._try_markdown_table(blocks):
                continue

            # 3. 尝试代码片段
            if self._try_code_block(blocks):
                continue
            # 4. 尝试图片
            if self._try_image(blocks):
                continue

            # 5. 尝试链接
            if self._try_link(blocks):
                continue

            # 没有匹配，移动到下一个字符
            self.pos += 1

        return blocks

    def _try_html_table(self, blocks):
        while self.pos < self.length and self.text[self.pos] in " \t\n\r":
            self.pos += 1

        if not self._match_literal("<table"):
            return False

        start = self.pos  # 保存当前位置（<table 开始）
        depth = 1
        self.pos += len("<table")

        while self.pos < self.length and depth > 0:
            if self._match_literal("<table"):
                depth += 1
                self.pos += len("<table")
                continue
            if self._match_literal("</table>"):
                depth -= 1
                self.pos += len("</table>")
                # 跳过零宽度空格和其他空白字符
                while self.pos < self.length and self.text[self.pos] in " \t\n\r\u200b\u00a0":
                    self.pos += 1
                continue
            self.pos += 1

        if depth == 0:
            blocks.append((start, self.pos, "html_table"))
            return True

        self.pos = start + len("<table")
        return False

    def _try_markdown_table(self, blocks):
        """尝试匹配 Markdown 表格"""
        # 跳过前导空白字符
        while self.pos < self.length and self.text[self.pos] in " \t\n\r":
            self.pos += 1

        start = self.pos

        line_start = self._get_line_start()
        if line_start is None:
            return False

        table_lines, end_pos = self._parse_markdown_table(line_start)

        if table_lines:
            self.pos = end_pos
            blocks.append((start, end_pos, "md_table"))
            return True

        return False

    def _try_code_block(self, blocks):
        # 跳过前导空白字符
        while self.pos < self.length and self.text[self.pos] in " \t\n\r":
            self.pos += 1

        start = self.pos

        if self._match_literal("```"):
            end_marker = "```"
        elif self._match_literal("~~~"):
            end_marker = "~~~"
        else:
            return False

        self.pos += len(end_marker)  # 跳过开始标记
        block_start = start

        while self.pos < self.length and self.text[self.pos] != "\n":
            self.pos += 1
        if self.pos < self.length:
            self.pos += 1

        while self.pos < self.length:
            if self._match_literal(end_marker):
                self.pos += len(end_marker)  # 跳过结束标记

                # 跳过零宽度空格和其他空白字符
                while self.pos < self.length and self.text[self.pos] in " \t\u200b\u00a0":
                    self.pos += 1

                blocks.append((block_start, self.pos, "code_block"))
                return True
            self.pos += 1

        blocks.append((block_start, self.pos, "code_block"))
        return True

    def _try_image(self, blocks):
        """尝试匹配图片（Markdown 或 HTML）"""
        start = self.pos

        # 跳过前导空白
        while self.pos < self.length and self.text[self.pos] in " \t\n\r":
            self.pos += 1

        # 格式1: Markdown 图片 ![alt](url)
        if self._match_literal("!["):
            self.pos += len("![")
            while self.pos < self.length and self.text[self.pos] != "]":
                self.pos += 1
            if self.pos < self.length:
                self.pos += 1
            if self.pos < self.length and self.text[self.pos] == "(":
                self.pos += 1
                while self.pos < self.length and self.text[self.pos] not in ")\n":
                    self.pos += 1
                if self.pos < self.length and self.text[self.pos] == ")":
                    self.pos += 1
                    blocks.append((start, self.pos, "image"))
                    return True

        # 格式3: HTML 图片 <img src="..." ... />
        self.pos = start
        if self._match_literal("<img"):
            while self.pos < self.length:
                if self._match_literal("/>"):
                    blocks.append((start, self.pos, "image"))
                    return True
                if self._match_literal(">"):
                    blocks.append((start, self.pos, "image"))
                    return True
                self.pos += 1

        self.pos = start
        return False

    def _try_link(self, blocks):
        """尝试匹配超链接（Markdown 或 HTML）"""
        start = self.pos

        # 跳过前导空白
        while self.pos < self.length and self.text[self.pos] in " \t\n\r":
            self.pos += 1

        # Markdown 链接 [text](url)
        if self._match_literal("["):
            self.pos += len("[")  # 跳过 [
            # 找到 link text
            while self.pos < self.length and self.text[self.pos] != "]":
                self.pos += 1
            if self.pos < self.length:
                self.pos += 1  # 跳过 ]
            if self.pos < self.length and self.text[self.pos] == "(":
                self.pos += 1  # 跳过 (
                # 找到 url
                while self.pos < self.length and self.text[self.pos] not in ")\n":
                    self.pos += 1
                if self.pos < self.length and self.text[self.pos] == ")":
                    self.pos += 1  # 跳过 )
                    blocks.append((start, self.pos, "link"))
                    return True

        # HTML 链接 <a href="...">text</a>
        self.pos = start
        if self._match_literal("<a"):
            while self.pos < self.length:
                if self._match_literal("</a>"):
                    blocks.append((start, self.pos, "link"))
                    return True
                self.pos += 1

        self.pos = start
        return False

    def _match_literal(self, literal):
        """检查当前位置是否匹配字面字符串"""
        if self.pos + len(literal) > self.length:
            return False
        return self.text[self.pos : self.pos + len(literal)] == literal

    def _get_line_start(self):
        """获取当前行的起始位置"""
        line_start = self.pos
        while line_start > 0 and self.text[line_start - 1] not in "\r\n":
            line_start -= 1
        return line_start

    def _parse_markdown_table(self, line_start):
        """解析 Markdown 表格，返回 (table_lines, end_pos)"""
        lines = []
        pos = line_start
        length = self.length

        # 先跳过开头的空行
        while pos < length:
            line_end = pos
            while line_end < length and self.text[line_end] not in "\r\n":
                line_end += 1
            line = self.text[pos:line_end].strip()
            if line == "":
                pos = line_end + 1
            else:
                break

        # 现在解析表格
        while pos < length:
            line_end = pos
            while line_end < length and self.text[line_end] not in "\r\n":
                line_end += 1

            line = self.text[pos:line_end].strip()

            # 检查是否是表格行（以 | 开头或包含 |）
            if not line.startswith("|") and "|" not in line:
                # 空行 - 说明表格结束，退出
                break

            # 检查是否是分隔行（只包含 |、-、:、空格）
            if self._is_separator_line(line):
                lines.append(line)
                pos = line_end + 1
                continue

            # 必须是有效的表格行
            if self._is_valid_table_row(line):
                lines.append(line)
            else:
                break

            pos = line_end + 1

        # 需要至少2行（表头+分隔行或表头+数据行）
        if len(lines) >= 2:
            return lines, pos

        return None, line_start

    def _is_separator_line(self, line):
        """检查是否是 Markdown 表格的分隔行"""
        stripped = line.strip(" |")
        return all(c in "-: " or c == "|" for c in stripped) and "-" in stripped

    def _is_valid_table_row(self, line):
        """检查是否是有效的表格行"""
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c]  # 移除空单元格
        return len(cells) >= 1


def extract_protected_blocks(text: str) -> List[tuple]:
    """提取文本中所有受保护的块

    Returns:
        List of (start, end, block_type) tuples
        block_type: 'html_table', 'md_table', 'code_block'
    """
    extractor = ProtectedBlockExtractor(text)
    return extractor.extract_all()


class MarkdownTextSplitter(RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter):
    """Text splitter for Markdown documents with hierarchical header support."""

    _HEADERS_TO_SPLIT_ON = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]

    @validate_params(
        chunk_size=dict(
            validator=lambda x: isinstance(x, int) and x > 0,
            message="chunk_size must be a positive integer",
        ),
        chunk_overlap=dict(
            validator=lambda x: isinstance(x, int) and x >= 0,
            message="chunk_overlap must be a non-negative integer",
        ),
        header_level=dict(
            validator=lambda x: isinstance(x, int) and 0 <= x <= 6,
            message="header_level must be an integer between 0 and 6",
        ),
    )
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 50,
        header_level: int = 3,
        **kwargs,
    ):
        """Initialize the text splitter with chunk size and overlap settings."""
        RecursiveCharacterTextSplitter.__init__(self, chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

        headers_to_split_on = self._HEADERS_TO_SPLIT_ON[:header_level]
        MarkdownHeaderTextSplitter.__init__(self, headers_to_split_on=headers_to_split_on)

    @validate_params(
        text=dict(
            validator=lambda x: isinstance(x, str) and len(x) <= MAX_SPLIT_SIZE,
            message="text must be a string, length range [0, 100 * 1024 * 1024]",
        )
    )
    def split_text(self, text: str) -> List[str]:
        """Split the input text into chunks according to size and header structure."""
        if len(text) <= self._chunk_size:
            return [text]

        # First split by headers
        header_chunks = MarkdownHeaderTextSplitter.split_text(self, text)

        result_chunks = []
        i = 0

        while i < len(header_chunks):
            current_chunk = header_chunks[i]
            accumulated_length = len(current_chunk.page_content)

            if accumulated_length > self._chunk_size:
                sub_chunks = self._split_with_protection(current_chunk.page_content)
                for sub_chunk in sub_chunks:
                    merged_content = self._merge_content_with_headers(
                        [Document(page_content=sub_chunk, metadata=current_chunk.metadata)],
                        0,
                        1,
                    )
                    result_chunks.append(merged_content)
                i += 1
                continue

            j = i + 1
            while (
                j < len(header_chunks) and accumulated_length + len(header_chunks[j].page_content) <= self._chunk_size
            ):
                accumulated_length = accumulated_length + len(header_chunks[j].page_content)
                j += 1

            merged_content = self._merge_content_with_headers(header_chunks, i, j)
            result_chunks.append(merged_content)
            i = j
        return result_chunks

    def _split_by_protected_blocks(self, text: str) -> List[tuple]:
        """
        Split text into segments of (content, is_protected) tuples.
        Protected blocks include tables and code blocks that should not be split.
        """
        blocks = extract_protected_blocks(text)

        segments = []
        last_end = 0

        for start, end, block_type in blocks:
            if start > last_end:
                non_protected_text = text[last_end:start].strip()
                if non_protected_text:
                    segments.append((non_protected_text, False))
            segments.append((text[start:end].rstrip("\r\n"), True))
            last_end = end

        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                segments.append((remaining, False))

        return segments

    def _split_with_protection(self, text: str) -> List[str]:
        """Split text while keeping protected blocks (tables, code) intact."""
        segments = self._split_by_protected_blocks(text)

        result = []
        current_parts = []
        current_length = 0

        for segment_text, is_protected in segments:
            segment_len = len(segment_text)

            if is_protected:
                if current_parts and current_length + segment_len > self._chunk_size:
                    result.append("\n\n".join(current_parts))
                    current_parts = []
                    current_length = 0
                current_parts.append(segment_text)
                current_length += segment_len
            else:
                if segment_len <= self._chunk_size:
                    if current_parts and current_length + segment_len > self._chunk_size:
                        result.append("\n\n".join(current_parts))
                        current_parts = []
                        current_length = 0
                    current_parts.append(segment_text)
                    current_length += segment_len
                else:
                    if current_parts:
                        result.append("\n\n".join(current_parts))
                        current_parts = []
                        current_length = 0
                    sub_chunks = RecursiveCharacterTextSplitter.split_text(self, segment_text)
                    result.extend(sub_chunks)

        if current_parts:
            result.append("\n\n".join(current_parts))

        return result

    def _merge_content_with_headers(self, header_chunks, sub_chunk_start, sub_chunk_end) -> str:
        """Merge content chunks with headers up from sub_chunk_start to sub_chunk_end."""
        parts = []
        prev_metadata = {}

        for idx in range(sub_chunk_start, sub_chunk_end):
            chunk = header_chunks[idx]
            metadata = chunk.metadata

            first_diff_level = None
            current_header = None
            for header_mark, header_name in self._HEADERS_TO_SPLIT_ON:
                if header_name in metadata and metadata[header_name]:
                    if current_header is None:
                        current_header = (header_mark, metadata[header_name])
                    if header_name not in prev_metadata or prev_metadata[header_name] != metadata[header_name]:
                        first_diff_level = header_name
                        break

            if first_diff_level is not None:
                output = False
                for header_mark, header_name in self._HEADERS_TO_SPLIT_ON:
                    if header_name == first_diff_level:
                        output = True
                    if output and header_name in metadata and metadata[header_name]:
                        level_num = int(header_name.split()[-1])
                        header_prefix = HEADER_MARK * level_num
                        parts.append(f"{header_prefix} {metadata[header_name]}")
            elif current_header and chunk.page_content.strip().startswith(("#", "|")):
                header_mark, header_text = current_header
                level_num = len(header_mark)
                header_prefix = HEADER_MARK * level_num
                parts.append(f"{header_prefix} {header_text}")

            if chunk.page_content:
                parts.append(chunk.page_content)
            prev_metadata = metadata

        return "\n\n".join(parts)
