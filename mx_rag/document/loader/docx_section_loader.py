# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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

from mx_rag.document.doc import Doc
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
        raise TypeError(f"TypeError {type(parent)}, should be Document")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


@dataclass
class ContentsHeading:
    title: str = ""
    sub_content: str = ""


class DocxLoaderByHead(DocxLoader):

    @staticmethod
    def _extract_hyperlink(block):
        try:
            hyperlink_rid = re.findall(r"<w:hyperlink r:id='(rId\d+)'", str(block.paragraph_format.element.xml))[0]
            return f" {block.part.rels[hyperlink_rid].target_ref} "
        except Exception as e:
            logger.warning(f"_extract_hyperlink {str(e)}")
            return ""

    @classmethod
    def _handle_paragraph_heading(cls, all_content: List[ContentsHeading], block: Paragraph,
                                  stack: List[Tuple[Any, Any]]) -> bool:
        """
        处理Heading级别元素，并将上级标题拼接到本级标题中
        """
        if block.style.name.startswith("Heading"):
            try:
                title_level = int(block.style.name.split()[-1])
            except Exception as ex:
                logger.warning(f"_handle_paragraph_heading {str(ex)}")
                return False
            while stack and stack[-1][0] >= title_level:
                stack.pop()
            if stack:
                parent_title = "".join(stack[-1][1])
                current_title = block.text
                block.text = parent_title + "-" + block.text
            else:
                current_title = block.text
            stack.append((title_level, current_title.strip()))
            all_content.append(ContentsHeading(block.text, ""))
            return True
        return False

    def load_and_split(self, text_splitter) -> List[Doc]:
        """
        将最小级别heading下的内容拼接生成Doc对象
        """
        if not self._is_document_valid():
            return []

        all_contents = [ContentsHeading()]
        stack = []

        doc = DocxDocument(self.file_path)
        for block in iter_block_items(doc):
            if isinstance(block, Table):
                res = self._handle_table(block)
                all_contents[-1].sub_content += res
            if not isinstance(block, Paragraph):
                logger.warning("skip current block")
                continue

            handle_head = self._handle_paragraph_heading(all_contents, block, stack)

            if block.style.name.lower().startswith("title"):
                all_contents[-1].title = block.text
                stack.append((0, block.text.strip()))
            elif not handle_head:
                all_contents[-1].sub_content += " " + block.text
                if "hyperlink" in block.paragraph_format.element.xml:
                    all_contents[-1].sub_content += self._extract_hyperlink(block)

        docs = []
        for content in all_contents:
            # 转化无意义特殊字符为标准字符
            plain_text = unicodedata.normalize("NFKD", content.sub_content).strip()
            # 过滤掉纯标题的document
            if len(plain_text) > 1:
                # 按定长切分进行分组
                grouped_text = text_splitter.split_text(plain_text)
                docs += [Doc(page_content=f"{unicodedata.normalize('NFKD', content.title).strip()} {text}",
                             metadata={"source": self.file_path}) for text in grouped_text]
        return docs
