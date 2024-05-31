# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
from typing import List

import fitz
from loguru import logger

from mx_rag.document.loader.document_loader import DocumentLoader, Doc
from mx_rag.utils import SecFileCheck


class PdfLoader(DocumentLoader):
    def __init__(self, file_path: str, image_inline=False):
        super().__init__(file_path)
        self.do_ocr = image_inline

    def load(self) -> List[Doc]:
        if not self._check():
            return []

        pdf_content = []
        pdf_document = fitz.open(self.file_path)
        docs: List[Doc] = list()

        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            pdf_content.append(page.get_text("text"))

        one_text = " ".join([t for t in pdf_content])
        docs.append(Doc(page_content=one_text, metadata={"source": self.file_path,
                                                         "page_count": pdf_document.page_count}))

        pdf_document.close()
        return docs

    def _get_pdf_page_count(self):
        pdf_document = fitz.open(self.file_path)
        pdf_page_count = pdf_document.page_count
        pdf_document.close()

        return pdf_page_count

    def _check(self):
        try:
            SecFileCheck(self.file_path, DocumentLoader.MAX_SIZE_MB).check()
            _pdf_page_count = self._get_pdf_page_count()
            if _pdf_page_count > DocumentLoader.MAX_PAGE_NUM:
                logger.error(f"too many pages {_pdf_page_count}")
                return False
            return True
        except Exception as e:
            logger.error(f"check file failed, {str(e)}")
            return False
