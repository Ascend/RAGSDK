# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
class Doc:
    page_content: str
    metadata: dict

    def __init__(self, page_content: str, metadata: dict) -> None:
        self.page_content = page_content
        self.metadata = metadata

    def __eq__(self, other):
        if not isinstance(other, Doc):
            return False
        return self.page_content == other.page_content

    def __lt__(self, other):
        return self.page_content < other.page_content

    def __hash__(self):
        return hash(self.page_content)
