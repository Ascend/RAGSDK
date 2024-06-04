# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.


__all__ = [
    "process_sentence",
    "DocxLoader",
    "ExcelLoader",
    "PdfLoader"
]

from mx_rag.document.loader.data_clean import process_sentence
from mx_rag.document.loader.docx_loader import DocxLoader
from mx_rag.document.loader.pdf_loader import PdfLoader
from mx_rag.document.loader.excel_loader import ExcelLoader
