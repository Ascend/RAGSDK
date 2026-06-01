# Demo运行说明

## 功能描述

以 API 方式调用 RAG SDK 构建知识库和问答系统的最基础样例，分为知识库构建和在线问答两个独立脚本。主要特性包括：

- **知识库构建**：支持多格式文档（txt、md、docx、pdf）的加载、切分、向量化和入库
- **在线问答**：支持向量检索 + reranker 精排 + LLM 生成的端到端问答
- **多线程并发**：支持多线程并行上传文档和并行问答
- **本地/服务化模型**：embedding 和 reranker 均支持本地部署或 TEI 服务化部署
- **Milvus 向量存储**：使用 Milvus 作为向量数据库，支持本地文件或服务化部署

## 前提条件

执行Demo前请先阅读[《RAG SDK 用户指南》](https://www.hiascend.com/document/detail/zh/mindsdk/730/rag/ragug/mxragug_0001.html)，并按照其中"安装部署"章节的要求完成必要软、硬件安装。
本章节为"应用开发"章节提供开发样例代码,便于开发者快速开发。

## 样例说明

详细的样例介绍请参考[《RAG SDK 用户指南》](https://www.hiascend.com/document/detail/zh/mindsdk/730/rag/ragug/mxragug_0001.html)"应用开发"章节说明。 其中：

> [!NOTE]
> 注意：创建知识库过程和在线问答过程使用的embedding模型、关系数据库路径、向量数据库路径需对应保持一致。其中关系数据库和向量数据库路径在样例代码中已经默认设置成一致，embedding模型需用户手动设置成一致。

## 运行及参数说明

1.调用示例

```commandline
# 上传知识库，支持多线程上传
python3 rag_demo_knowledge.py  --file_path "/home/data/MindIE.docx" --file_path "/home/data/gaokao.docx"

# 在线问答，支持多线程问答
python3 rag_demo_query.py --query "请描述2024年高考作为题目" --query "请问2025年一共有多少天法定节假日"
```

> [!NOTE]
> 说明: 调用示例前请先根据用户实际情况完成参数配置,确保embedding模型路径正确，大模型能正常访问，文件路径正确等，参数可以通过修改样例代码，也可通过命令行的方式传入。

2.参数说明

```commandline
#以"创建知识库"为例,用户可以通过以下命令查看参数情况;如需开发其他样例,请详细参考《RAG SDK用户指南》"接口参考"章节。
python3 rag_demo_knowledge.py  --help
```
