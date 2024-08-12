# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple, Any, Iterator

import docx
from docx.document import Document as mxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from loguru import logger
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

from mx_rag.document.doc import Doc
from mx_rag.document.loader.base_loader import BaseLoader as mxBaseLoader
from mx_rag.utils.file_check import SecFileCheck



@dataclass
class ContentsHeading:
    title: str = ""
    sub_content: str = ""


class DocxLoader(BaseLoader, mxBaseLoader):
    """Loading logic for loading documents from docx."""
    EXTENSION = (".docx",)

    def __init__(self, file_path: str, image_inline=False):
        """Initialize with filepath and options."""
        super().__init__(file_path)
        self.do_ocr = image_inline
        self.table_index = 0

    @staticmethod
    def extract_hyperlink(block):
        try:
            hyperlink_rid = re.findall(r"<w:hyperlink r:id='(rId\d+)'", str(block.paragraph_format.element.xml))[0]
            return f" {block.part.rels[hyperlink_rid].target_ref} "
        except Exception as e:
            logger.warning(f"extract_hyperlink {str(e)}")
            return ""

    @staticmethod
    def iter_block_items(parent: mxDocument):
        """
        获取Document对象的元素
        按文档顺序生成对*parent*中每个段落和表的引用。
        每个返回值都是Table或Paragraph。
        *parent*通常是对主Document对象的引用。
        """
        if isinstance(parent, Document):
            parent_elm = parent.element.body
        else:
            raise TypeError(f"TypeError {type(parent)}, should be mxDocument")

        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)

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

    def lazy_load(self) -> Iterator[Document]:
        """Load documents."""
        if not self._is_document_valid():
            yield Document(page_content='')

        all_text = []
        doc = docx.Document(self.file_path)
        for element in doc.element.body:
            if element.tag.endswith("tbl"):
                table_text = self._handle_table(doc.tables[self.table_index])
                self.table_index += 1
                all_text.append(table_text)
            elif element.tag.endswith("p"):
                if "pic:pic" in str(element.xml) and self.do_ocr:
                    logger.debug("pic:pic set to empty str")
                    pic_texts = ""
                    all_text.extend(pic_texts)
                paragraph = docx.text.paragraph.Paragraph(element, doc)
                para_text = paragraph.text
                all_text.append(para_text)

        one_text = "\n\n".join([t for t in all_text])
        yield Document(page_content=one_text, metadata={"source": os.path.basename(self.file_path)})


    def _handle_table(self, element):
        """docx.oxml.table.CT_Tbl"""

        logger.info(f"handle docx table {self.table_index}")
        rows = list(element.rows)
        headers = [cell.text for cell in rows[0].cells]
        data = [[cell.text.replace("\n", " ") for cell in row.cells] for row in rows[1:]]
        result = ["，".join([f"{x}: {y}" for x, y in zip(headers, subdata)]) for subdata in data]
        res = "；".join(result)
        return res + "。"

    def _is_document_valid(self):
        try:
            SecFileCheck(self.file_path, self.MAX_SIZE).check()
            if not self.file_path.endswith(DocxLoader.EXTENSION):
                logger.error(f"file type not correct")
                return False
            if self._is_zip_bomb():
                logger.error(f"file too large")
                return False

            doc = docx.Document(self.file_path)
            if len(doc.paragraphs) > self.MAX_PAGE_NUM:
                logger.error(f"too many pages {len(doc.paragraphs)}")
                return False
            return True
        except Exception as e:
            logger.error(f"check file failed, {str(e)}")
            return False
