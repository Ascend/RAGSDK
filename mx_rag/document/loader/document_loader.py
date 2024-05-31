from abc import ABC, abstractmethod


class Doc:
    page_content: str
    metadata: dict

    def __init__(self, page_content: str, metadata: dict) -> None:
        self.page_content = page_content
        self.metadata = metadata


class DocumentLoader(ABC):
    MAX_SIZE_MB = 100
    MAX_PAGE_NUM = 1000

    def __init__(self, file_path):
        self.file_path = file_path

    @abstractmethod
    def load(self):
        pass
