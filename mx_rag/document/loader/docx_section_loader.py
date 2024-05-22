#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import re
from dataclasses import dataclass
from typing import List, Tuple, Any

import unicodedata
from docx import Document as DocxDocument
from docx.document import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from loguru import logger

from mx_rag.document.loader.docx_loader import Doc
from mx_rag.document.loader.docx_loader import DocxLoader


def iter_block_items(parent: Document):
    """
    获取Document对象的元素
    按文档顺序生成对*parent*中每个段落和表子的引用。
    每个返回值都是Table或Paragraph。
    *parent*通常是对主Document对象的引用。
    """
    if isinstance(parent, Document):
        parent_elm = parent.element.body
    else:
        raise TypeError(f"对象类型{type(parent)}错误, 应为Document类型")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


@dataclass
class ContentsHeading:
    title: str = ''
    sub_content: str = ''


class DocxLoaderByHead(DocxLoader):

    @staticmethod
    def _extract_hyperlink(block):
        try:
            hyperlink_rid = re.findall(r'<w:hyperlink r:id="(rId\d+)"', str(block.paragraph_format.element.xml))[0]
            logger.debug(f"hyperlink_rid is {hyperlink_rid}, block.text is {block.text}")
            return f' {block.part.rels[hyperlink_rid].target_ref} '
        except Exception as e:
            logger.info(f"_extract_hyperlink {str(e)}, {block.paragraph_format.element.xml}")
            return ""

    @classmethod
    def _handle_paragraph_heading(cls, all_content: List[ContentsHeading], block: Paragraph,
                                  stack: List[Tuple[Any, Any]]) -> bool:
        """
        处理Heading级别元素，并将上级标题拼接到本级标题中
        """
        if block.style.name.startswith('Heading'):
            try:
                title_level = int(block.style.name.split()[-1])
                logger.info(f"title_level is {block.style.name} -> {title_level}")
            except Exception as ex:
                logger.debug(f"_handle_paragraph_heading {str(ex)}")
                return False
            logger.debug(f"stack is {stack}")
            while stack and stack[-1][0] >= title_level:
                stack.pop()
            if stack:
                parent_title = ''.join(stack[-1][1])
                current_title = block.text
                block.text = parent_title + '-' + block.text
                logger.warning(
                    f"modify block text {block.text}: parent_title={parent_title}, current_title={current_title}")
            else:
                current_title = block.text
            stack.append((title_level, current_title.strip()))
            all_content.append(ContentsHeading(block.text, ''))
            return True
        return False

    def load_and_split(self, text_splitter) -> List[Document]:
        """
        将最小级别heading下的内容拼接生成Doc对象
        """

        all_contents = [ContentsHeading()]
        stack = []

        doc = DocxDocument(self.doc_path)
        for block in iter_block_items(doc):
            logger.debug(f"block is {block}")
            if isinstance(block, Table):
                res = self.handle_table(block)
                all_contents[-1].sub_content += res
            if not isinstance(block, Paragraph):
                logger.warning("skip current block")
                continue

            handle_head = self._handle_paragraph_heading(all_contents, block, stack)

            if block.style.name.lower().startswith('title'):
                logger.warning(f"block style is title {block.style.name.lower()}")
                all_contents[-1].title = block.text
                stack.append((0, block.text.strip()))
            elif not handle_head:
                all_contents[-1].sub_content += " " + block.text
                if 'hyperlink' in block.paragraph_format.element.xml:
                    all_contents[-1].sub_content += self._extract_hyperlink(block)

        docs = []
        for content in all_contents:
            logger.debug(f"final process {content}")
            # 转化无意义特殊字符为标准字符
            plain_text = unicodedata.normalize('NFKD', content.sub_content).strip()
            # 过滤掉纯标题的document
            if len(plain_text) > 1:
                # 按定长切分进行分组
                grouped_text = text_splitter.split_text(plain_text)
                logger.debug(f"grouped_text is {grouped_text}")
                docs += [Doc(page_content=f"{unicodedata.normalize('NFKD', content.title).strip()} {text}",
                             metadata={'source': self.doc_path}) for text in grouped_text]
        return docs
