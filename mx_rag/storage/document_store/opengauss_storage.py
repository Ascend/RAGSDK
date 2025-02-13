# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from typing import List, Optional, Callable
from sqlalchemy import URL

from mx_rag.storage.document_store.base_storage import MxDocument, Docstore
from mx_rag.storage.document_store.helper_storage import HelperDocStore
from mx_rag.utils.common import validate_params, MAX_CHUNKS_NUM


class OpenGaussDocstore(Docstore):
    def __init__(self, url: URL, encrypt_fn: Callable = None, decrypt_fn: Callable = None):
        super().__init__()
        self.doc_store = HelperDocStore(url, encrypt_fn, decrypt_fn)

    @validate_params(documents=dict(
        validator=lambda x: 0 < len(x) <= MAX_CHUNKS_NUM and all(isinstance(it, MxDocument) for it in x),
        message="param must be List[MxDocument] and length range in (0, 1000 * 1000]"))
    def add(self, documents: List[MxDocument], document_id: int) -> List[int]:
        return self.doc_store.add(documents, document_id)

    def delete(self, document_id: int) -> List[int]:
        return self.doc_store.delete(document_id)

    @validate_params(chunk_id=dict(validator=lambda x: x >= 0, message="param must greater equal than 0"))
    def search(self, chunk_id: int) -> Optional[MxDocument]:
        return self.doc_store.search(chunk_id)

    def get_all_index_id(self) -> List[int]:
        return self.doc_store.get_all_index_id()
