# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from pathlib import Path
from typing import Iterator
from loguru import logger
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from mx_rag.utils import file_check
from mx_rag.document.loader.base_loader import BaseLoader as mxBaseLoader


class ImageLoader(BaseLoader, mxBaseLoader):
    def __init__(self, file_path):
        super().__init__(file_path)

    def lazy_load(self) -> Iterator[Document]:
        """
        ：返回：逐行读取表,返回 string list
        """
        try:
            file_check.excel_file_check(self.file_path, self.MAX_SIZE)
        except Exception as e:
            logger.error(e)
            yield Document(page_content='')
        file = Path(self.file_path)
        yield Document(page_content=file.as_posix(), metadata={"path": file.as_posix()})