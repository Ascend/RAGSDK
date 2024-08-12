# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.


__all__ = [
    "DocxLoader",
    "ExcelLoader",
    "PdfLoader",
    "PowerPointLoader",
    "BaseLoader"

]

from mx_rag.document.loader.ppt_loader import PowerPointLoader
from mx_rag.document.loader.docx_loader import DocxLoader
from mx_rag.document.loader.pdf_loader import PdfLoader
from mx_rag.document.loader.excel_loader import ExcelLoader
from mx_rag.document.loader.ppt_loader import PowerPointLoader
from mx_rag.document.loader.base_loader import BaseLoader

