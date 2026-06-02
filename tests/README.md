# RAG SDK 测试指南

- [简介](#简介)
- [测试类型说明](#测试类型说明)
- [环境准备](#环境准备)
- [全量测试](#全量测试)
- [单文件测试](#单文件测试)
- [用例编写指南](#用例编写指南)

# 简介

本文档为 RAG SDK 项目测试指南，说明项目中 UT（单元测试）与 ST（前冒烟测试）的划分、运行方式及用例编写规范。

- **UT（Unit Test）**：位于 `tests/unit_tests/` 目录，通过 mock 隔离外部依赖，在无真实服务环境下即可运行，用于验证各模块的逻辑正确性。
- **ST（Presmoke Test）**：位于 `tests/presmoke/` 目录，需要部署真实服务（Milvus、Embedding 模型、LLM 等），在 PR 门禁流水线中作为最后一道关卡，验证代码合入后基本功能不受影响。
也可参照[Ascend社区开发者测试贡献指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/developer-testing-guide.md)下分类目录中单元测试和系统测试介绍。

# 测试类型说明

## UT（单元测试）

| 特征 | 说明 |
|:--|:--|
| 目录 | `tests/unit_tests/` |
| 框架 | pytest + unittest.TestCase |
| 外部依赖 | 使用 `unittest.mock`（MagicMock、patch）隔离，无需真实服务 |
| 覆盖率 | 通过 pytest-cov 采集，配置见根目录 `.coveragerc.txt` |
| 执行时机 | 每次构建时自动执行，也支持本地手动运行 |
| 数据文件 | 使用 `tests/data/` 下的公共测试数据 |

## ST（前冒烟测试）

| 特征 | 说明 |
|:--|:--|
| 目录 | `tests/presmoke/` |
| 框架 | pytest + unittest.TestCase |
| 外部依赖 | 需要真实服务（Milvus、Embedding 模型、LLM 推理服务等），或使用 `emb_model_service.py` 提供的 mock 服务 |
| 执行时机 | PR 门禁流水线中自动触发，根据变更文件按映射关系选择执行对应 presmoke 用例 |
| 映射规则 | 见 `presmoke/map_presmoke_list.py` 中的 `change_st_mapping` 字典 |

# 环境准备

1. 安装依赖：

    ```bash
    pip install pytest pytest-cov pytest-html
    pip install -r ../requirements.txt
    ```

2. 设置环境变量（`run_py_test.sh` 中已包含）：

    ```bash
    export PYTHONPATH=<项目根目录>:<tests/fake_package路径>:$PYTHONPATH
    ```

3. ST 额外要求：
   - Milvus 服务可访问（默认地址 `http://my-release-milvus.milvus:19530`）
   - Embedding 模型已部署到 `/home/data/bge-large-zh-v1.5`
   - LLM 推理服务已启动（默认地址 `http://127.0.0.1:8000/v1/chat/completions`）
   - 或者启动 mock 服务：`python tests/presmoke/emb_model_service.py`

# 全量测试

## 运行全量 UT

使用项目提供的脚本一键执行所有 UT 并生成覆盖率报告：

```bash
cd tests
bash run_py_test.sh
```

该脚本会：

- 设置 `PYTHONPATH` 指向项目根目录和 `fake_package`
- 执行 `tests/unit_tests/` 下所有 UT 用例，排除 `tests/presmoke/`
- 生成覆盖率报告（HTML + XML）到 `test_results/` 目录

等效的 pytest 命令：

```bash
python3 -m pytest \
    --cov=mx_rag \
    --cov-report=html \
    --cov-report=xml \
    --junit-xml=./final.xml \
    --html=./final.html \
    --self-contained-html \
    --durations=5 \
    -vs \
    --cov-branch \
    --cov-config=.coveragerc \
    --ignore=tests/presmoke/* \
    tests/unit_tests/
```

## 运行全量 ST

ST 用例需要在已部署服务的环境中执行：

```bash
python3 -m pytest -vs tests/presmoke/
```

# 单文件测试

## 运行单个 UT 文件

```bash
python3 -m pytest -vs tests/unit_tests/cache/test_cache_core.py
```

## 运行单个 ST 文件

```bash
python3 -m pytest -vs tests/presmoke/knowledge/test_ragsdk_demo.py
```

## 运行单个测试类或方法

```bash
# 运行某个测试类
python3 -m pytest -vs tests/unit_tests/cache/test_cache_core.py::TestCacheCore

# 运行某个测试方法
python3 -m pytest -vs tests/unit_tests/cache/test_cache_core.py::TestCacheCore::test_core_init
```

# 用例编写指南

## UT 用例编写

UT 位于 `tests/unit_tests/` 目录，按模块分子目录存放。编写规范：

### 文件命名

- 文件名以 `test_` 开头，如 `test_cache_core.py`
- 与被测模块对应，如 `mx_rag/cache/` → `tests/unit_tests/cache/test_cache_core.py`

### 关键原则

1. **隔离外部依赖**：使用 `unittest.mock.patch` 或 `MagicMock` 隔离 Milvus、模型服务等外部依赖
2. **使用 fake_package**：对于 faiss、paddleocr 等难以 mock 的依赖，`tests/fake_package/` 提供了桩实现，通过 `PYTHONPATH` 自动加载
3. **测试数据**：使用 `tests/data/` 下的公共测试数据，通过相对路径引用：

   ```python
   current_dir = os.path.dirname(os.path.realpath(__file__))
   test_file = os.path.realpath(os.path.join(current_dir, "../../data/test.md"))
   ```

4. **清理临时文件**：在 `setUp` 中创建、`tearDown` 中清理，避免影响其他用例
5. **每个 test 方法验证一个独立场景**，避免单个方法内断言过多

## ST（前冒烟）用例编写

ST 位于 `tests/presmoke/` 目录，按模块分子目录存放。

### 文件命名

- 文件名以 `test_` 开头，如 `test_ragsdk_demo.py`
- 放置在对应模块的 presmoke 子目录下

### 编写规范

```python
import os
import unittest

from mx_rag.embedding.service import TEIEmbedding
from mx_rag.knowledge import KnowledgeDB, KnowledgeStore
from mx_rag.utils import ClientParam
# ... 其他导入


class TestXxxPresmoke(unittest.TestCase):
    def setUp(self):
        # 清理残留数据
        if os.path.exists("./sql.db"):
            os.remove("./sql.db")

    def test_end_to_end(self):
        # 配置真实服务地址
        embedding_url = "http://127.0.0.1:8000/v1/embeddings"
        milvus_url = "http://my-release-milvus.milvus:19530"
        # ... 完整的端到端流程测试
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
```

### 关键原则

1. **使用真实服务**：ST 用例需要连接真实的 Milvus、Embedding、LLM 等服务
2. **使用公共测试数据**：通过 `tests/data/` 下的文件，路径使用相对定位：

   ```python
   file_path = os.path.realpath(os.path.join(
       os.path.dirname(os.path.realpath(__file__)), "../../data/gaokao.txt"))
   ```

3. **清理环境**：`setUp` 中清理 `sql.db` 等残留文件，`tearDown` 中还原环境
4. **添加映射关系**：新增 ST 用例后，需在 `presmoke/map_presmoke_list.py` 的 `change_st_mapping` 中注册变更文件到用例的映射

## 添加映射关系

当新增 ST 用例时，需要在 `tests/presmoke/map_presmoke_list.py` 中更新 `change_st_mapping` 字典，将源码变更路径映射到对应的 presmoke 用例：

```python
# 单文件映射
"mx_rag/cache/cache_generate_qas/html_makrdown_parser.py": "tests/presmoke/cache/test_markdown_parser.py",

# 目录映射（目录下任意文件变更都会触发对应目录下所有用例）
"mx_rag/cache": "tests/presmoke/cache",

# 单文件映射到多个用例
"mx_rag/chain/single_text_to_text.py": [
    "tests/presmoke/knowledge/test_ragsdk_demo.py",
    "tests/presmoke/graph/test_graph_pipline.py"
],
```

匹配规则为**最长前缀匹配**：遍历 `change_st_mapping` 的 key，找到与变更文件路径最长匹配的 key，返回其对应的 presmoke 用例。若无匹配，默认执行 `tests/presmoke/knowledge/test_ragsdk_demo.py`。
