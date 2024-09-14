# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import concurrent
import re
from abc import abstractmethod, ABC
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ssl import SSLContext
from typing import List, Any, Tuple, Dict

from langchain_community.document_loaders import TextLoader
from loguru import logger

from mx_rag.utils.common import validate_params, validata_list_str
from mx_rag.utils.file_check import FileCheck, SecFileCheck
from mx_rag.utils.url import RequestUtils

MAX_FILE_SIZE_10M = 10 * 1024 * 1024
MAX_FILE_NUM = 1000


class GenerateQaParser(ABC):

    @abstractmethod
    def parse(self):
        pass


def _thread_pool_callback(worker):
    worker_exception = worker.exception()
    if worker_exception:
        logger.error(
            "called thread pool executor callback function, worker return exception: {}".format(worker_exception))


class HTMLParser(GenerateQaParser):
    """
    功能描述:
        这是一个用于解析HTML页面的类，它继承自GenerateQaParser类。
        它使用多线程技术，通过HTTP请求获取HTML页面，并使用readability库提取页面的标题和内容。
    Attributes:
        urls: 需要解析的HTML页面的URL列表
        headers: HTTP请求的头部信息
        timeout: HTTP请求的超时时间，默认为10秒
        cert_file: SSL证书文件的路径，默认为空
        crl_file: SSL证书撤销列表文件的路径，默认为空
        use_http: 是否使用HTTP协议，默认为False
        proxy_url: 代理服务器的URL，默认为空
        ssl_context: SSL上下文对象，默认为None
        max_url_num: 最大解析url数量
    """
    @validate_params(
        urls=dict(validator=lambda x: validata_list_str(x, [1, 10000], [1, 1000])),
        headers=dict(validator=lambda x: x is None or isinstance(x, dict)),
        timeout=dict(validator=lambda x: isinstance(x, int) and 0 < x < 1000),
        use_http=dict(validator=lambda x: isinstance(x, bool)),
        ssl_context=dict(validator=lambda x: x is None or isinstance(x, SSLContext))
    )
    def __init__(self, urls: List[str], headers: Dict = None, timeout: int = 10, cert_file: str = "",
                 crl_file: str = "", use_http: bool = False, proxy_url: str = "", ssl_context: SSLContext = None):
        if headers is None:
            headers = {'Content-Type': 'application/json'}
        self.urls = urls
        self.headers = headers
        self._client = RequestUtils(timeout=timeout, cert_file=cert_file, crl_file=crl_file,
                                    use_http=use_http, proxy_url=proxy_url, ssl_context=ssl_context)

    def parse(self) -> Tuple[List[str], List[str]]:
        def _request(client: RequestUtils, url: str):
            import readability
            from html_text import html_text
            response = client.get(url, self.headers)
            if not response:
                return "", ""
            html_doc = readability.Document(response)
            title = html_doc.title()
            content = html_text.extract_text(html_doc.summary(html_partial=True))
            if not title or not content:
                logger.warning(f"Failed to get title or content of '{url}', skip")
            return title, content

        titles = []
        contents = []
        task_list = []
        with ThreadPoolExecutor() as executor:
            for url in self.urls:
                thread_pool_exc = executor.submit(
                    _request,
                    self._client,
                    url
                )
                thread_pool_exc.add_done_callback(_thread_pool_callback)
                task_list.append(thread_pool_exc)
        for future in concurrent.futures.as_completed(task_list):
            title, content = future.result()
            if not title or not content:
                continue
            titles.append(title)
            contents.append(content)
        return titles, contents


def _md_load(file_path: str) -> List[str]:
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    docs = []
    for document in documents:
        # 过滤markdown中以base64编码的图片内容
        content = re.sub(r"^.*data:image.*$", "", document.page_content, flags=re.I | re.M)
        docs.append(content)
    return docs


class MarkDownParser(GenerateQaParser):
    """
    功能描述:
        这是一个用于解析markdown的类，它继承自GenerateQaParser类。
        返回内容为文件名和内容
    Attributes:
        file_path: 需要解析的markdown所在文件夹
        max_file_num: 解析的最大文件数
    """
    @validate_params(
        max_file_num=dict(validator=lambda x: isinstance(x, int) and 1 <= x <= 10000)
    )
    def __init__(self, file_path: str, max_file_num: int = MAX_FILE_NUM):
        FileCheck.dir_check(file_path)
        FileCheck.check_files_num_in_directory(file_path, ".md", max_file_num)
        self.file_path = file_path

    def parse(self) -> Tuple[List[str], List[str]]:
        def _load_file(_mk: Any):
            SecFileCheck(_mk.as_posix(), MAX_FILE_SIZE_10M).check()
            for doc in _md_load(_mk.as_posix()):
                return _mk.name, doc

        titles = []
        contents = []
        task_list = []

        with ThreadPoolExecutor() as executor:
            for _mk in Path(self.file_path).glob("*.md"):
                thread_pool_exc = executor.submit(
                    _load_file,
                    _mk
                )
                thread_pool_exc.add_done_callback(_thread_pool_callback)
                task_list.append(thread_pool_exc)
        for future in concurrent.futures.as_completed(task_list):
            title, content = future.result()
            titles.append(title)
            contents.append(content)
        return titles, contents
