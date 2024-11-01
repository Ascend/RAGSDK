# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import base64
import os
from pathlib import Path
from typing import Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from mx_rag.document.loader.base_loader import BaseLoader as mxBaseLoader
from mx_rag.utils.common import MAX_PAGE_CONTENT
from mx_rag.utils.file_check import SecFileCheck

IMAGE_TYPE = (".jpg", ".png")


class ImageLoader(BaseLoader, mxBaseLoader):
    def __init__(self, file_path):
        super().__init__(file_path)

    def lazy_load(self) -> Iterator[Document]:
        """
        ：返回：逐行读取表,返回 string list
        """
        # 图片不做切分，最大取值和入库MxDocument page_content大小保持一致
        SecFileCheck(self.file_path, MAX_PAGE_CONTENT).check()
        if Path(self.file_path).suffix not in IMAGE_TYPE:
            raise TypeError(f"type '{Path(self.file_path).suffix}' is not support")

        with open(self.file_path, "rb") as fi:
            encode_content = str(base64.b64encode(fi.read()).decode())

        yield Document(page_content=encode_content, metadata={"source": os.path.basename(self.file_path)})
