# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import inspect
from pathlib import Path
from typing import Callable, List, Tuple, Dict

import numpy as np
from loguru import logger

from mx_rag.document import SUPPORT_DOC_TYPE, SUPPORT_IMAGE_TYPE
from mx_rag.knowledge import KnowledgeDB
from mx_rag.utils import FileCheck


class FileHandlerError(Exception):
    def __init__(self, err_msg: str):
        self.err_msg = err_msg
        info: inspect.FrameInfo = inspect.stack()[1]
        msg = f"{info.function}({Path(info.filename).name}:{info.lineno})-{err_msg}"
        super().__init__(msg)


def upload_files(
        knowledge: KnowledgeDB,
        files: List[str],
        parse_func: Callable[[str], Tuple[List[str], List[Dict[str, str]]]],
        embed_func: Callable[[List[str]], np.ndarray],
        force: bool = False
        ):
    """上传单个文档，不支持的文件类型会抛出异常，如果文档重复，可选择强制覆盖"""
    if len(files) > knowledge.max_loop_limit:
        raise FileHandlerError(f'files list length must less than {knowledge.max_loop_limit}, upload files failed')

    for file in files:
        FileCheck.check_path_is_exist_and_valid(file)
        file_obj = Path(file)

        in_white_path = False
        for p in knowledge.white_paths:
            if file_obj.resolve().is_relative_to(p):
                in_white_path = True
        if not in_white_path:
            raise FileHandlerError(f"{file_obj.as_posix()} is not in whitelist path")

        if not file_obj.is_file():
            raise FileHandlerError(f"{file} is not a normal file")

        if knowledge.check_document_exist(file_obj.name):
            if not force:
                raise FileHandlerError(f"file path {file_obj.name} is already exist")
            else:
                knowledge.delete_file(file_obj.name)

        texts, metadatas = parse_func(file_obj.as_posix())
        try:
            knowledge.add_file(file_obj.name, texts, embed_func, metadatas)
        except Exception as err:
            # 当添加文档失败时，删除已添加的部分文档做回滚，捕获异常是为了正常回滚
            try:
                knowledge.delete_file(file_obj.name)
            except Exception as e:
                logger.warning(f"exception encountered while rollback, {e}")
            raise FileHandlerError(f"add {file_obj.name} failed, {err}") from err


def upload_dir(
        knowledge: KnowledgeDB,
        dir_path: str,
        parse_func: Callable[[str], Tuple[List[str], List[Dict[str, str]]]],
        embed_func: Callable[[List[str]], np.ndarray],
        force=False,
        load_image=False
):
    """
    只遍历当前目录下的文件，不递归查找子目录文件，目录中不支持的文件类型会跳过，如果文档重复，可选择强制覆盖，超过最大文件数量则退出
    load_image为True时导入支持的类型图片, False时支持导入支持的文档
    """
    dir_path_obj = Path(dir_path)
    if not dir_path_obj.is_dir():
        raise FileHandlerError(f"dir path {dir_path} is invalid")

    support_file_type = SUPPORT_DOC_TYPE
    if load_image:
        support_file_type = SUPPORT_IMAGE_TYPE

    count = 0
    files = []
    for file in Path(dir_path).glob("*"):
        if count >= knowledge.max_loop_limit:
            logger.warning("the number of files reaches the maximum limit")
            break
        if file.suffix in support_file_type:
            files.append(file.as_posix())
        count += 1

    upload_files(knowledge, files, parse_func, embed_func, force)


def delete_files(
        knowledge: KnowledgeDB,
        file_names: List[str]
):
    """删除上传的文档，需传入待删除的文档名称"""
    if not isinstance(file_names, list) or not file_names:
        raise FileHandlerError(f"files param {file_names} is invalid")

    count = 0
    for filename in file_names:
        if not isinstance(filename, str):
            raise FileHandlerError(f"file path {filename} is invalid")
        if count >= knowledge.max_loop_limit:
            logger.warning("the number of files reaches the maximum limit")
            break
        if not knowledge.check_document_exist(filename):
            continue
        knowledge.delete_file(filename)
        count += 1
