# 前置知识

在使用 RAG SDK 之前，建议您了解以下基本概念和操作，以便更好地理解和使用本产品。

## RAG 基本概念

| 概念 | 说明 | 了解更多 |
|:--|:--|:--|
| RAG（检索增强生成） | 通过为大模型接入外部知识库，提升问答系统准确率，解决大模型幻觉和时效性问题 | [Wikipedia](https://zh.wikipedia.org/wiki/%E6%AA%A2%E7%B4%A2%E5%A2%9E%E5%BC%B7%E7%94%9F%E6%88%90) |
| Embedding（向量化） | 将文本、图像等数据转换为向量表示，便于相似度计算和检索 | [产品介绍](./01-introduction.md) |
| 向量数据库 | 存储向量数据的专用数据库，支持高效的相似性检索 | [数据库说明](./api/databases.md) |
| Reranker（重排序） | 对检索结果进行精细化排序，提高最终返回结果的质量 | [排序模块](./api/reranker.md) |
| 知识库 | 存储和管理领域知识的结构化数据集合 | [知识管理](./api/knowledge_management.md) |

## 硬件与环境要求

使用 RAG SDK 前，请确保您已了解以下环境信息：

- **昇腾 NPU 硬件**：Atlas 300I Duo 推理卡、Atlas 800I A2/A3 推理服务器等
  - 了解 NPU 驱动和固件的安装与版本管理
  - 参见：[固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community)
- **操作系统**：支持 Ubuntu、openEuler、KylinOS 等 Linux 发行版
- **Docker 容器**：推荐使用容器化部署，需了解 Docker 的基本使用
  - [Docker 官方文档](https://docs.docker.com/)
- **CANN 软件栈**：昇腾 AI 处理器的软件运行支撑平台
  - [CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html)

## 相关技术知识

| 领域 | 建议掌握内容 | 参考资源 |
|:--|:--|:--|
| Python | 基础语法、pip 包管理 | [Python 官方文档](https://docs.python.org/3/) |
| 大语言模型 | LLM 基本概念、API 调用方式 | [Qwen3 文档](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Dense.html) |
| Milvus | 向量数据库的基本概念与使用 | [Milvus 文档](https://milvus.io/docs/zh) |
| LangChain | RAG 应用开发框架 | [LangChain 文档](https://python.langchain.com/) |

## 快速导航

如果您已了解上述知识，可以直接进入以下章节：

- [快速入门](./03-quickstart.md)：5 分钟完成从部署到示例运行
- [安装部署](./04-installation_guide.md)：详细的安装指导
- [开发流程](./05-user_guide.md)：完整的应用开发指南
