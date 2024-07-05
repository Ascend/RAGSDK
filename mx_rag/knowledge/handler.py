# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import json
import os
import stat
from pathlib import Path
from typing import Callable, List, Tuple, Dict

import numpy as np
from loguru import logger
from transformers import PreTrainedTokenizerBase

from mx_rag.document import SUPPORT_DOC_TYPE, SUPPORT_IMAGE_TYPE
from mx_rag.knowledge.base_knowledge import KnowledgeBase
from mx_rag.knowledge.knowledge import KnowledgeTreeDB, KnowledgeDB
from mx_rag.retrievers.tree_retriever.tree_structures import Tree, Node, tree2dict
from mx_rag.utils import FileCheck, SecFileCheck


class FileHandlerError(Exception):
    pass


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
        check_file(file, force, knowledge)
        file_obj = Path(file)
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


def check_file(file: str, force: bool, knowledge: KnowledgeBase):
    """
    检查文件路径
    """
    FileCheck.check_path_is_exist_and_valid(file)
    file_obj = Path(file)
    if not is_in_white_paths(file_obj, knowledge.white_paths):
        raise FileHandlerError(f"{file_obj.as_posix()} is not in whitelist path")
    if not file_obj.is_file():
        raise FileHandlerError(f"{file} is not a normal file")
    if knowledge.check_document_exist(file_obj.name):
        if not force:
            raise FileHandlerError(f"file path {file_obj.name} is already exist")
        else:
            knowledge.delete_file(file_obj.name)


def is_in_white_paths(file_obj: Path, white_paths: List[str]) -> bool:
    """
    判断是否在白名单路径中
    """
    for p in white_paths:
        if file_obj.resolve().is_relative_to(p):
            return True
    return False


def upload_files_build_tree(knowledge: KnowledgeTreeDB,
                            files: List[str],
                            parse_func: Callable[[str, PreTrainedTokenizerBase, int], Tuple],
                            embed_func: Callable[[List[str]], np.ndarray],
                            force: bool = False) -> Tree:
    if len(files) > 1:
        raise FileHandlerError(f"Currently not supported for uploading multiple files simultaneously!")
    for file in files:
        check_file(file, force, knowledge)
    tokenizer = knowledge.tree_builder_config.tokenizer
    max_tokens = knowledge.tree_builder_config.max_tokens
    total_texts = []
    total_metadatas = []
    file_names = []
    for file in files:
        file_obj = Path(file)
        texts, metadatas = parse_func(file_obj.as_posix(), tokenizer, max_tokens)
        total_texts.extend(texts)
        total_metadatas.extend(metadatas)
        [file_names.append(file_obj.name) for i in range(len(texts))]
    return knowledge.add_files(file_names, total_texts, embed_func, total_metadatas)


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


def save_tree(tree: Tree, file_path: str):
    """
    序列化Tree并保存
    """
    if tree is None:
        raise ValueError("There is no tree to save.")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(file_path, flags, modes), "w") as ff:
        ff.write(json.dumps(tree, default=tree2dict))


def load_tree(file_path: str, white_paths, float_type=np.float16):
    """
    从文件加载Tree，反序列化
    """
    real_path = os.path.realpath(file_path)
    file_obj = Path(real_path)
    if not is_in_white_paths(file_obj, white_paths):
        raise FileHandlerError(f"{file_obj.as_posix()} is not in whitelist path")
    file_check = SecFileCheck(file_path, 1024 * 1024 * 1024)
    file_check.check()
    with open(file_path, "r") as f:
        tree_dict = json.load(f)
    all_nodes = json2node(tree_dict.get("all_nodes"), float_type)
    root_nodes = json2node(tree_dict.get("root_nodes"), float_type)
    leaf_nodes = json2node(tree_dict.get("leaf_nodes"), float_type)
    layer_to_nodes = josn2node_list(tree_dict.get("layer_to_nodes"), float_type)
    num_layers = tree_dict.get("num_layers")
    return Tree(all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes)


def json2node(dict_nodes: List[Dict[str, Dict]], float_type) -> {}:
    """
    反序列化Node对象
    """
    result = {}
    for item in dict_nodes:
        for k, v in item.items():
            children = set(v.get("children", []))
            embeddings = np.array(v.get("embeddings", []), dtype=float_type)
            index = int(v.get("index", 0))
            text = v.get("text", "")
            result[int(k)] = Node(text, index, children, embeddings)
    return result


def josn2node_list(dict_node_list: List[Dict[str, List[Dict[str, str]]]], float_type):
    result = {}
    for item in dict_node_list:
        for k, v in item.items():
            node_list = []
            for node in v:
                children = set([int(child) for child in node.get("children", [])])
                embeddings = np.array(node.get("embeddings", []), dtype=float_type)
                index = int(node.get("index", 0))
                text = node.get("text", "")
                node_list.append(Node(text, index, children, embeddings))
            result[int(k)] = node_list
    return result
