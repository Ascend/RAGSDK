# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from typing import List, Iterator

from pathlib import Path
import fitz
import cv2
import numpy as np
from loguru import logger

from paddleocr import PPStructure
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
from PIL import Image

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

from mx_rag.document.loader.base_loader import BaseLoader as mxBaseLoader
from mx_rag.utils.file_check import SecFileCheck
from mx_rag.utils.common import validate_params, BOOL_TYPE_CHECK_TIP, Lang


class PdfLoader(BaseLoader, mxBaseLoader):
    EXTENSION = (".pdf",)
    
    @validate_params(
        lang=dict(validator=lambda x: isinstance(x, Lang), message="param must be instance of Lang"),
        layout_recognize=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
    )
    def __init__(self, file_path: str, lang: Lang = Lang.EN, layout_recognize: bool = False):
        super().__init__(file_path)
        self.layout_recognize = layout_recognize
        self.ocr_engine = None
        self.lang = lang

    @staticmethod
    def _reconstruct(layout_res):
        pdf_content: List[str] = []
        for page_layout in layout_res:
            for line in page_layout:
                PdfLoader._reconstruct_line(line, pdf_content)
        return pdf_content

    @staticmethod
    def _reconstruct_line(line, pdf_content):
        line.pop('img')
        for res in line['res']:
            if 'text' in res:
                pdf_content.append(res['text'])
        pdf_content.append("\n")

    def lazy_load(self) -> Iterator[Document]:
        self._check()
        return self._parser() if self.layout_recognize else self._plain_parser()

    def _text_merger(self, pdf_content):
        one_text = " ".join(pdf_content)
        yield Document(page_content=one_text, metadata={"source": self.file_path,
                                                        "page_count": self._get_pdf_page_count()})

    def _layout_recognize(self, pdf_document):
        layout_res = []
        imgs = []

        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            mat = fitz.Matrix(2, 2)

            # 获取页面的宽高（未放大）
            rect = page.rect
            estimated_width = int(rect.width * 2)  # 放大两倍后的宽度
            estimated_height = int(rect.height * 2)  # 放大两倍后的高度
            # 判断放大两倍后是否超出4096像素
            if estimated_width > 4096 or estimated_height > 4096:
                raise ValueError(f"Page {page_num} 2*size exceed limit 4096 : width={rect.width},"
                                 f" height={rect.height}")

            pm = page.get_pixmap(matrix=mat, alpha=False)

            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            imgs.append(img)
            del img

        for img in imgs:
            ocr_res = self.ocr_engine(img)
            h, w, _ = img.shape
            result = sorted_layout_boxes(ocr_res, w)
            layout_res.append(result)

        return layout_res

    def _parser(self):
        if self.ocr_engine is None:
            try:
                self.ocr_engine = PPStructure(table=True, ocr=True, lang=self.lang.value, layout=True, show_log=False)
            except AssertionError as e:
                logger.error(f"Assertion error: {e}")
                self.ocr_engine = None
                return self._text_merger([""])
            except Exception as e:
                logger.error(f"paddleOcr init failed, {e}")
                self.ocr_engine = None
                return self._text_merger([""])

        with fitz.open(self.file_path) as pdf_document:
            layout_res = self._layout_recognize(pdf_document)

        pdf_content = self._reconstruct(layout_res)

        return self._text_merger(pdf_content)

    def _plain_parser(self):
        pdf_content = []

        with fitz.open(self.file_path) as pdf_document:
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                pdf_content.append(page.get_text("text"))

        return self._text_merger(pdf_content)

    def _get_pdf_page_count(self):
        pdf_document = fitz.open(self.file_path)
        pdf_page_count = pdf_document.page_count
        pdf_document.close()

        return pdf_page_count

    def _check(self):
        SecFileCheck(self.file_path, self.MAX_SIZE).check()
        if not self.file_path.endswith(PdfLoader.EXTENSION):
            raise TypeError(f"type '{Path(self.file_path).suffix}' is not support")
        _pdf_page_count = self._get_pdf_page_count()
        if _pdf_page_count > self.MAX_PAGE_NUM:
            raise ValueError(f"too many pages {_pdf_page_count}")