# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import List, Literal

import docx
from loguru import logger
from pydantic import Field

from mx_rag.utils import SecFileCheck


class Doc:
    """Class for storing a piece of text and associated metadata."""

    page_content: str
    metadata: dict = Field(default_factory=dict)
    type: Literal["Doc"] = "Doc"

    def __init__(self, page_content: str, metadata: dict) -> None:
        """Pass page_content in as positional or named arg."""
        self.page_content = page_content
        self.metadata = metadata


class DocxLoader:
    """Loading logic for loading documents from docx."""

    def __init__(self, file_path: str, image_inline=False, max_size_mb=100):
        """Initialize with filepath and options."""
        self.doc_path = file_path
        self.do_ocr = image_inline
        self.table_index = 0
        self.max_size_mb = max_size_mb

    def load(self) -> List[Doc]:
        """Load documents."""

        if not self._check():
            return []

        docs: List[Doc] = list()
        all_text = []
        doc = docx.Document(self.doc_path)
        for element in doc.element.body:
            if element.tag.endswith("tbl"):
                table_text = self._handle_table(doc.tables[self.table_index])
                self.table_index += 1
                logger.debug(f"table: {table_text}")
                all_text.append(table_text)
            elif element.tag.endswith("p"):
                if "pic:pic" in str(element.xml) and self.do_ocr:
                    logger.debug("pic:pic set to empty str")
                    pic_texts = ''
                    all_text.extend(pic_texts)
                paragraph = docx.text.paragraph.Paragraph(element, doc)
                logger.debug(f"paragraph: {paragraph.text}")
                para_text = paragraph.text
                all_text.append(para_text)

        one_text = " ".join([t for t in all_text])
        docs.append(Doc(page_content=one_text, metadata={"source": self.doc_path}))
        return docs

    def _handle_table(self, element):
        """docx.oxml.table.CT_Tbl"""

        logger.info(f"handle docx table {self.table_index}")
        rows = list(element.rows)
        headers = [cell.text for cell in rows[0].cells]
        data = [[cell.text.replace('\n', ' ') for cell in row.cells] for row in rows[1:]]
        result = ['，'.join([f"{x}: {y}" for x, y in zip(headers, subdata)]) for subdata in data]
        res = '；'.join(result)
        return res + '。'

    def _check(self):
        try:
            SecFileCheck(self.doc_path, self.max_size_mb).check()
            return True
        except Exception as e:
            logger.error(f"check file failed, {str(e)}")
            return False
