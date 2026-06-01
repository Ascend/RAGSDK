# Demo运行说明

## 功能描述

基于 RAG SDK 缓存模块的问答样例，演示如何使用多级缓存加速 RAG 问答。主要特性包括：

- **多级缓存**：支持 memory cache（L1）和 similarity cache（L2）两级缓存架构
- **完全一致缓存**：memory cache 支持精确匹配，命中时直接返回
- **语义相似缓存**：similarity cache 支持语义相似度匹配，使用 embedding 和 reranker 进行相似度计算
- **Milvus 向量存储**：使用 Milvus 作为向量数据库，支持本地文件或服务化部署

## 前提条件

执行Demo前请先阅读[《RAG SDK 用户指南》](https://www.hiascend.com/document/detail/zh/mindsdk/730/rag/ragug/mxragug_0001.html)，并按照其中"安装部署"章节的要求完成必要软、硬件安装。
本章节为"应用开发"章节提供开发样例代码,便于开发者快速开发。

## 样例说明

详细的样例介绍请参考[《RAG SDK 用户指南》](https://www.hiascend.com/document/detail/zh/mindsdk/730/rag/ragug/mxragug_0001.html)"应用开发"章节说明。 其中：

> [!NOTE]
> 注意：创建知识库过程和在线问答过程使用的embedding模型、关系数据库路径、向量数据库路径需对应保持一致。其中关系数据库和向量数据库路径在样例代码中已经默认设置成一致，embedding模型需用户手动设置成一致。

## 运行及参数说明

1. 上传知识文档示例。

```commandline
# 上传知识库
python3 cache_demo_knowledge.py  --embedding_url http://127.0.0.1:8080/v1/embeddings --file_path /home/1.md --file_path /home/2.txt
```

2.知识问答示例

相同问题调用两次，可通过日志观察到耗时差异，首次调用耗时较长，后续调用耗时较短。

```commandline
# 上传知识库
python3 cache_demo_query.py --embedding_url http://127.0.0.1:8080/v1/embeddings --reranker_url http://127.0.0.1:8081/v1/rerank --llm_url http://127.0.0.1:1025/v1/chat/completions --model_name Llama3-8B-Chinese-Chat --query "RAG架构是怎样的"
```

## 注意事项

调用示例前请先根据用户实际情况完成参数配置,确保embedding、reranker、llm模型服务能正确访问，文件路径正确等。
