# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import List, Literal

import docx
from loguru import logger
from pydantic import Field

from mx_rag.document.loader.document_loader import DocumentLoader, Doc
from mx_rag.utils import SecFileCheck


class DocxLoader(DocumentLoader):
    """Loading logic for loading documents from docx."""

    def __init__(self, file_path: str, image_inline=False):
        """Initialize with filepath and options."""
        super().__init__(file_path)
        self.do_ocr = image_inline
        self.table_index = 0

    def load(self) -> List[Doc]:
        """Load documents."""

        if not self._is_document_valid():
            return []

        docs: List[Doc] = list()
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
                    pic_texts = ''
                    all_text.extend(pic_texts)
                paragraph = docx.text.paragraph.Paragraph(element, doc)
                para_text = paragraph.text
                all_text.append(para_text)

        one_text = " ".join([t for t in all_text])
        docs.append(Doc(page_content=one_text, metadata={"source": self.file_path}))
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

    def _is_document_valid(self):
        try:
            SecFileCheck(self.file_path, DocumentLoader.MAX_SIZE_MB).check()
            doc = docx.Document(self.file_path)
            if len(doc.paragraphs) > DocumentLoader.MAX_PAGE_NUM:
                logger.error(f"too many pages {len(doc.paragraphs)}")
                return False
            return True
        except Exception as e:
            logger.error(f"check file failed, {str(e)}")
            return False
