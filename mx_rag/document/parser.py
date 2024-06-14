# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from pathlib import Path
from typing import List, Dict, Tuple

from mx_rag.document.loader import DocxLoader, ExcelLoader, PdfLoader
from mx_rag.document.splitter import CharTextSplitter


DOC_PARSER_MAP = {
    ".docx": (DocxLoader, CharTextSplitter),
    ".xlsx": (ExcelLoader, CharTextSplitter),
    ".xls": (ExcelLoader, CharTextSplitter),
    ".csv": (ExcelLoader, CharTextSplitter),
    ".pdf": (PdfLoader, CharTextSplitter),
}
SUPPORT_IMAGE_TYPE = (".jpg", ".png")


def parse_file(filepath: str) -> Tuple[List[str], List[Dict[str, str]]]:

    def parse_image(file: Path) -> Tuple[List[str], List[Dict[str, str]]]:
        return [file.as_posix()], [{"path": file.as_posix()}]

    def parse_document(file: Path) -> Tuple[List[str], List[Dict[str, str]]]:
        loader, splitter = DOC_PARSER_MAP.get(file.suffix)
        metadatas = []
        texts = []
        for doc in loader(file.as_posix()).load():
            split_texts = splitter().split_text(doc.page_content)
            metadatas.extend(doc.metadata for _ in split_texts)
            texts.extend(split_texts)
        return texts, metadatas

    file_obj = Path(filepath)
    if file_obj.suffix in DOC_PARSER_MAP.keys():
        texts, metadatas = parse_document(file_obj)
    elif file_obj.suffix in SUPPORT_IMAGE_TYPE:
        texts, metadatas = parse_image(file_obj)
    else:
        raise ValueError(f"{file_obj.suffix} is not support")
    return texts, metadatas
