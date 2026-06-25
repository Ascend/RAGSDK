# Demo运行说明

## 功能描述

本样例面向教培行业数学课程知识问答场景，提供一套轻量可执行的 RAG 最佳实践参考。样例覆盖从课程资料清洗、数学公式规范化、知识切分、检索到 Prompt 组装的基本流程，并给出对接 RAGSDK 现有能力的扩展方式。

主要特性包括：

- **数学资料清洗**：统一全角/半角符号，保留 `x^2 - 5x + 6 = 0` 等公式上下文，降低 OCR 后的检索噪声。
- **章节化切分**：按 Markdown 标题切分课程资料，再使用 `chunk_size` 和 `chunk_overlap` 保留解题步骤连续性。
- **本地可运行检索**：内置一个仅依赖 Python 标准库的 TF-IDF 检索器，用于验证最佳实践流程。
- **生产扩展路径**：可替换为 RAGSDK 的 embedding、reranker、OCR/VLM、知识库管理和模型服务能力。

## 前提条件

本地验证仅需 Python 3.9 及以上版本，不依赖外部模型服务。

生产环境部署时，请先阅读仓库内 26.0.0 配套文档[《RAG SDK 用户指南》](../../docs/zh/user_guide.md)和[《安装部署》](../../docs/zh/installation_guide.md)，完成 RAGSDK、LLM、embedding、reranker、OCR/VLM 等服务部署。

## 最佳实践流程

### 1. 资料采集与 OCR

教培数学资料通常来自 PDF、图片、PPT 或 Word。建议按以下顺序处理：

1. 对扫描版资料先进行 OCR，保留页码、章节标题、例题编号等结构信息。
2. 对公式进行人工或规则校验，重点检查上标、分式、根号、正负号和括号。
3. 将资料整理为 Markdown 或结构化 JSON，便于后续按章节和例题切分。

如需图文并茂问答，可参考 `example/chat_with_ascend` 中 OCR/VLM 服务的部署和调用方式。

### 2. 公式规范化

OCR 后常见问题包括全角符号、空格异常、公式边界丢失等。建议在入库前完成基础规范化：

- 将 `（ ） ＋ － ＝ × ÷` 等全角或特殊符号统一为半角表达。
- 保留公式所在标题、例题和解题步骤，避免只保存单行公式。
- 对容易混淆的字符进行抽检，例如 `0/O`、`1/l`、`x/×`。

本样例在 `normalize_math_text` 中演示了基础符号规范化逻辑。

### 3. 知识切分

数学问答的 chunk 不宜只按固定长度切分。推荐策略：

- 优先按章节、知识点、例题、步骤分层。
- 初始 `chunk_size` 可设置为 500 到 800 中文字符，`chunk_overlap` 可设置为 80 到 150 字符。
- 如果资料以短例题为主，可减小 `chunk_size`，避免多个无关题目进入同一片段。
- 如果资料包含长证明或连续推导，可适当增加 overlap，保证检索片段包含完整公式上下文。

本地 demo 默认使用较小的 `chunk_size=260`，便于在样例资料中展示多个片段。

### 4. Embedding 模型选择与微调

建议先使用通用中文 embedding 模型建立召回基线，再基于垂域数据微调：

1. 使用课程讲义、例题解析、错题讲解生成 query-document 或 query-answer 训练样本。
2. 覆盖定义问答、公式检索、步骤推理、易错点解释等问题类型。
3. 使用 Recall@k、命中片段质量和人工抽检评估微调收益。

RAGSDK 已提供向量模型垂域微调样例，可参考 `example/embedding_finetune`。

### 5. 问答生成

生产环境中，建议将召回片段、题目、公式约束和回答格式一起组装 Prompt：

- 要求模型只依据给定资料回答。
- 要求公式保持规范写法。
- 对计算题要求分步骤回答。
- 对资料不足的问题明确说明无法从知识库确认。

本样例的 `build_prompt` 展示了 Prompt 组装方式。

## 运行 Demo

进入样例目录：

```bash
cd example/education_math_rag
```

运行内置问题：

```bash
python3 education_math_rag_demo.py
```

指定问题并输出 Prompt：

```bash
python3 education_math_rag_demo.py \
  --question "一元二次方程 x^2 - 5x + 6 = 0 的两个解是什么？" \
  --show_prompt
```

调整切分参数：

```bash
python3 education_math_rag_demo.py \
  --chunk_size 320 \
  --chunk_overlap 60 \
  --top_k 3
```

执行内置自检：

```bash
python3 education_math_rag_demo.py --self_check
```

预期输出会展示：

- 加载的知识片段数量
- 每个问题召回的 Top K 片段
- 可选的 RAG Prompt
- 自检模式下，每个内置问题的 Top 1 片段应匹配预期知识点

## 对接 RAGSDK 生产能力

本样例中的 `LocalTfidfRetriever` 仅用于本地可执行验证。生产环境建议替换为以下能力：

| 本地 demo 步骤 | 生产环境建议 |
| --- | --- |
| `normalize_math_text` | 在 OCR 后处理链路中加入公式和符号规范化规则 |
| `split_by_markdown_heading` / `split_text_with_overlap` | 结合 RAGSDK 文档解析和 splitter 能力，按章节、例题、步骤切分 |
| `LocalTfidfRetriever` | 替换为 RAGSDK embedding + vector store + reranker |
| `build_prompt` | 接入 LLM 服务，增加资料不足、公式格式和分步解答约束 |
| `data/math_lesson.md` | 替换为真实教培课程资料和标注问答对 |

相关参考：

- `example/chat_with_ascend`：图形化知识问答系统，包含 OCR/VLM、Milvus、embedding/reranker 服务对接。
- `example/embedding_finetune`：向量模型垂域微调端到端流程。
- `docs/zh/api/embedding_model_fine_tuning.md`：embedding 微调 API 说明。
