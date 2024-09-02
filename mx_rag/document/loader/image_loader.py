# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import base64
import os
from typing import Iterator
from loguru import logger
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from mx_rag.document.loader.base_loader import BaseLoader as mxBaseLoader
from mx_rag.utils.file_check import SecFileCheck


class ImageLoader(BaseLoader, mxBaseLoader):
    def __init__(self, file_path):
        super().__init__(file_path)

    def lazy_load(self) -> Iterator[Document]:
        """
        ：返回：逐行读取表,返回 string list
        """
        try:
            SecFileCheck(self.file_path, self.MAX_SIZE).check()
        except Exception as e:
            logger.error(e)
            yield Document(page_content='')

        with open(self.file_path, "rb") as fi:
            encode_content = str(base64.b64encode(fi.read()).decode())

        yield Document(page_content=encode_content, metadata={"source": os.path.basename(self.file_path)})
