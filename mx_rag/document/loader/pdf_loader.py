# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
from typing import List

import fitz
from loguru import logger

from mx_rag.utils import SecFileCheck


class Doc:
    page_content: str
    metadata: dict

    def __init__(self, page_content: str, metadata: dict) -> None:
        self.page_content = page_content
        self.metadata = metadata


class PdfLoader:
    def __init__(self, file_path: str, image_inline=False, max_size_mb=100, max_page_num=1000):
        self.pdf_path = file_path
        self.do_ocr = image_inline
        self.max_size_mb = max_size_mb
        self.max_page_num = max_page_num

    def load(self) -> List[Doc]:
        if not self._check():
            return []

        pdf_content = []
        pdf_document = fitz.open(self.pdf_path)
        docs: List[Doc] = list()

        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            pdf_content.append(page.get_text("text"))

        one_text = " ".join([t for t in pdf_content])
        docs.append(Doc(page_content=one_text, metadata={"source": self.pdf_path,
                                                         "page_count": pdf_document.page_count}))

        pdf_document.close()
        return docs

    def _get_pdf_page_count(self):
        pdf_document = fitz.open(self.pdf_path)
        pdf_page_count = pdf_document.page_count
        pdf_document.close()

        return pdf_page_count

    def _check(self):
        try:
            SecFileCheck(self.pdf_path, self.max_size_mb).check()
            _pdf_page_count = self._get_pdf_page_count()
            if _pdf_page_count > self.max_page_num:
                logger.error(f"too many pages {_pdf_page_count}")
                return False
            return True
        except Exception as e:
            logger.error(f"check file failed, {str(e)}")
            return False
