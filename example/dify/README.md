# Demo运行样例

## 功能描述

RAG SDK 提供知识库对接开源 Dify 框架的样例。通过 FastAPI 服务将 RAG SDK 的知识库管理、文档检索、问答能力封装为标准接口，供 Dify 平台调用。主要特性包括：

- **Dify 外接知识库**：作为 Dify 平台的外部知识库服务，提供 `/retrieval` 和 `/query` 接口
- **文档管理 API**：提供文档上传（`/upload_file`）、删除（`/delete_file`）、列表查询（`/list_files`）等接口
- **图文并茂回答**：可选启用 VLM 模型解析文档中的图片，生成图文交错回答
- **混合检索**：支持向量检索和 BM25 全文检索，结果经 reranker 精排后返回

## 前提条件

执行Demo前请先阅读[《RAG SDK 用户指南》](https://www.hiascend.com/document/detail/zh/mindsdk/730/rag/ragug/mxragug_0001.html)，并按照其中"安装部署"章节的要求完成必要软、硬件安装。
本章节为"应用开发"章节提供开发样例代码,便于开发者快速开发。

## 配套版本说明

| 服务 | 版本要求 | 说明 |
|:--|:--|:--|
| Dify | 0.15.3 | 开源 LLM 应用开发平台 |
| Milvus | v2.5.0 及以上 | 向量数据库 |
| mis-tei embedding/reranker | - | 昇腾 embedding 和 reranker 服务化部署 |
| LLM 服务 | - | 大语言模型推理服务 |
| VLM 服务（可选） | - | 视觉语言模型，用于图片解析 |

## 使用步骤

1. 部署milvus服务（[部署参考链接](https://milvus.io/docs/zh/install_standalone-docker.md)）

2. 部署mis-tei emb，reranker服务（[部署参考链接](https://www.hiascend.com/developer/ascendhub/detail/07a016975cc341f3a5ae131f2b52399d)）

3. 如果需要解析docx、pdf文件中的图片进行图文并茂回答，启动demo时请配置 --parse_image 使能图片解析功能，需部署VLM模型服务（[部署参考链接](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen-VL-Dense.html)），LLM服务（[部署参考链接](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Dense.html)），注意如果图片尺寸长或宽小于256，由于信息小，将被丢弃处理

4. 执行dify_demo.py运行服务,具体参数可执行 --help查看

    ```bash
    python3 dify_demo.py --host "${HOST_IP}" --port "${PORT}" --llm_base_url "${LLM_BASE_URL}" --vlm_base_url "${VLM_BASE_URL}" --embedding_url "${EMBEDDING_URL}" --reranker_url "${RERANKER_URL}"
    ```

    > [!NOTE]
    >- HOST_IP、PORT为用户服务的IP和端口信息
    >- LLM_BASE_URL 为默认或者用户配置的LLM大模型服务地址
    >- VLM_BASE_URL 为默认或者用户配置的VLM大模型服务地址
    >- EMBEDDING_URL 为默认或者用户配置的向量模型服务地址
    >- RERANKER_URL 为默认或者用户配置的排序模型服务地址
    >- 若用户需要通过网页访问服务的doc文档，需要在PORT后拼接/docs，访问示例：http://{HOST_IP}:{PORT}/docs

5. 通过接口上传、删除、查看文档等操作

6. 支持在dify界面配置外接知识库，[部署参考链接](https://docs.dify.ai/zh-hans/guides/knowledge-base/connect-external-knowledge-base)

7. 可调用/query接口问答测试,代码执行路径下存放了LLM回答文件response.md，可通过如下代码启动web网页可直观展示答复内容，复制如下代码在dify_demo.py同级目录下创建st.py

```python
import streamlit as st
import re

# 读取 Markdown 文件
with open("./response.md", "r", encoding="utf-8") as file:
    markdown_text = file.read()

# 在 Streamlit 应用中显示 Markdown 内容，同时处理图片
def render_markdown_with_images(markdown_text):
    # 匹配 Markdown 图片语法 ![alt text](image_url)
    pattern = re.compile(r'!\[.*?\]\((.*?)\)')

    # 记录上一个位置
    last_pos = 0

    # 查找所有匹配项
    for match in pattern.finditer(markdown_text):
        # 显示上一个位置到匹配位置之间的文本
        st.markdown(markdown_text[last_pos:match.start()], unsafe_allow_html=True)

        # 显示图片
        img_url = match.group(1)
        st.image(img_url)

        # 更新上一个位置
        last_pos = match.end()

    # 显示剩余的文本
    st.markdown(markdown_text[last_pos:], unsafe_allow_html=True)

render_markdown_with_images(markdown_text)
```

## 调用函数显示内容

这里只是简单提供展示样例，如果需考虑安全，请开启https安全认证功能。

WEB服务启动命令：

```bash
streamlit run st.py --server.address "${HOST_IP}" --server.port "${PORT}"
```
