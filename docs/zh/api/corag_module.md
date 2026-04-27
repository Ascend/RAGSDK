# CoRAG接口参考

## CoRAG模块

CoRAG（Chain of Retrieval-Augmented Generation）是一种基于链式检索增强生成的多轮问答框架，通过迭代生成子查询、获取相关文档并整合信息，实现复杂问题的深度推理和回答。

### CoRagBaseConfig

#### 类功能

**功能描述**

CoRAG基础配置类，包含共享的核心参数，用于初始化CoRAG相关组件。

**函数原型**

```python
from mx_rag.corag.config import CoRagBaseConfig
CoRagBaseConfig(base_llm, retrieve_api_url, num_threads, max_path_length, final_llm, sub_answer_llm, judge_llm)
```

**参数说明**

| 参数名                | 数据类型         | 可选/必选 | 默认值  | 说明                                                                                       |
| ------------------ | ------------ | ----- | ---- | ---------------------------------------------------------------------------------------- |
| base\_llm          | Text2TextLLM | 必选    | -    | 基础LLM实例，用于生成子查询和答案。具体可参见[Text2TextLLM](./llm_client.md#text2textllm)                     |
| retrieve\_api\_url | str          | 必选    | -    | 检索API的URL地址，用于获取相关文档，必须支持POST请求，请求体为JSON格式，包含查询文本query，返回体为JSON格式，支持多种结构（详见下方请求体和响应体示例）。 |
| num\_threads       | int          | 可选    | 8    | 并行处理的线程数。                                                                                |
| max\_path\_length  | int          | 可选    | 3    | 最大路径长度，表示生成子查询的最大轮数。                                                                     |
| final\_llm         | Text2TextLLM | 可选    | None | 最终答案生成LLM实例，若不提供则使用base\_llm。具体可参见[Text2TextLLM](./llm_client.md#text2textllm)           |
| sub\_answer\_llm   | Text2TextLLM | 可选    | None | 子答案生成LLM实例，若不提供则使用base\_llm。具体可参见[Text2TextLLM](./llm_client.md#text2textllm)            |
| judge\_llm         | Text2TextLLM | 可选    | None | 判断LLM实例，用于评估答案正确性。具体可参见[Text2TextLLM](./llm_client.md#text2textllm)                      |
| retrieve\_top\_k   | int          | 可选    | 5    | 检索API返回的文档数量，默认值为5。                                                                          |
| client\_param     | ClientParam  | 可选    | ClientParam() | HTTP客户端参数，用于配置请求行为。具体可参见[ClientParam](./universal_api.md#clientparam) |

**请求体示例**

```json
{
    "query": "Which company acquired by Google was founded first?", "top_k": 5
}
```

**响应体示例**

支持多种响应格式，以下是常见的几种：

#### 格式1：包含document\_ids和documents的标准格式

```json
{
    "document_ids": ["doc1", "doc2"],
    "documents": ["Google is a multinational technology company.", "YouTube was founded on February 14, 2005."]
}
```

#### 格式2：包含chunks字段的格式

```json
{
    "chunks": [
        "Google is a multinational technology company that specializes in Internet-related services and products.",
        "YouTube is an American online video sharing and social media platform owned by Google."
    ]
}
```

#### 格式3：包含data字段的格式

```json
{
    "data": [
        {
            "id": "doc1",
            "content": "Google is a multinational technology company."
        },
        {
            "id": "doc2",
            "content": "YouTube was founded on February 14, 2005."
        }
    ]
}
```

#### 格式4：包含results字段的格式

```json
{
    "results": [
        {
            "doc_id": "doc1",
            "text": "Google is a multinational technology company that specializes in Internet-related services and products."
        },
        {
            "doc_id": "doc2",
            "text": "YouTube is an American online video sharing and social media platform owned by Google."
        }
    ]
}
```

#### 格式5：包含docs字段的格式

```json
{
    "docs": [
        {
            "id": "doc1",
            "contents": "Google is a multinational technology company."
        },
        {
            "id": "doc2",
            "contents": "YouTube was founded on February 14, 2005."
        }
    ]
}
```

#### 格式6：包含passages字段的格式

```json
{
    "passages": [
        {
            "id": "doc1",
            "content": "Google is a multinational technology company."
        },
        "YouTube was founded on February 14, 2005."
    ]
}
```

**支持的响应字段说明**：

- 文档内容可从以下字段提取：`content`, `contents`, `text`
- 文档ID可从以下字段提取：`id`, `doc_id`
- 支持直接返回字符串列表或包含上述字段的字典列表

### ReasoningPath

#### 类功能

**功能描述**

表示CoRAG推理路径的数据类，包含原始查询、子查询、子答案、文档ID、思考和文档的列表。

**函数原型**

```python
from mx_rag.corag.corag_agent import ReasoningPath
ReasoningPath(original_query, subqueries, subanswers, document_ids, reasoning_steps, documents)
```

**参数说明**

| 参数名              | 数据类型              | 可选/必选 | 默认值 | 说明                    |
| ---------------- | ----------------- | ----- | --- | --------------------- |
| original\_query  | str               | 必选    | -   | 原始查询文本。               |
| subqueries       | List\[str]        | 可选    | \[] | 子查询列表。                |
| subanswers       | List\[str]        | 可选    | \[] | 子答案列表。                |
| document\_ids    | List\[List\[str]] | 可选    | \[] | 文档ID列表，每个子查询对应多个文档ID。 |
| reasoning\_steps | List\[str]        | 可选    | \[] | 思考过程列表，每个子查询对应一个思考过程。 |
| documents        | List\[List\[str]] | 可选    | \[] | 文档内容列表，每个子查询对应多个文档内容。 |

### CoRagAgent

#### 类功能

**功能描述**

CoRAG智能体类，负责生成推理路径和最终答案，是CoRAG框架的核心组件。

**函数原型**

```python
from mx_rag.corag.corag_agent import CoRagAgent
CoRagAgent(base_llm, retrieve_api_url, final_llm, sub_answer_llm)
```

**参数说明**

| 参数名                | 数据类型         | 可选/必选 | 默认值  | 说明                                                                             |
| ------------------ | ------------ | ----- | ---- | ------------------------------------------------------------------------------ |
| base\_llm          | Text2TextLLM | 必选    | -    | 基础LLM实例，用于生成子查询和答案。具体可参见[Text2TextLLM](./llm_client.md#text2textllm)           |
| retrieve\_api\_url | str          | 必选    | -    | 检索API的URL地址，用于获取相关文档。                                                          |
| final\_llm         | Text2TextLLM | 可选    | None | 最终答案生成LLM实例，若不提供则使用base\_llm。具体可参见[Text2TextLLM](./llm_client.md#text2textllm) |
| sub\_answer\_llm   | Text2TextLLM | 可选    | None | 子答案生成LLM实例，若不提供则使用base\_llm。具体可参见[Text2TextLLM](./llm_client.md#text2textllm)  |
| retrieve\_top\_k   | int          | 可选    | 5    | 检索API返回的文档数量，默认值为5。                                                                          |
| client\_param     | ClientParam  | 可选    | ClientParam() | HTTP客户端参数，用于配置请求行为。具体可参见[ClientParam](./universal_api.md#clientparam) |

#### 调用示例

```python
from mx_rag.corag.corag_agent import CoRagAgent
from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils import ClientParam

# 初始化LLM实例
llm = Text2TextLLM(base_url="https://{ip}:{port}/v1/chat/completions",
                   model_name="qianwen-7b",
                   llm_config=LLMParameterConfig(max_tokens=512),
                   client_param=ClientParam(ca_file="/path/to/ca.crt")
                   )

# 初始化CoRagAgent
agent = CoRagAgent(
    base_llm=llm,
    retrieve_api_url="http://your-retrieve-api.com/retrieve",
    retrieve_top_k=5,
    client_param=ClientParam(ca_file="/path/to/ca.crt")
)

# 生成推理路径
task_desc = "回答用户的复杂问题，通过多轮子查询获取相关信息"
rag_path = agent.sample_path(
    query="什么是CoRAG框架的工作原理？",
    task_desc=task_desc,
    max_path_length=3
)

# 生成最终答案
final_answer = agent.generate_final_answer(
    rag_path=rag_path,
    task_description=task_desc
)

print("最终答案:", final_answer)
```

#### sample\_path

**功能描述**

通过迭代生成子查询，根据子查询从数据源检索相关文档，并收集子答案和相关文档，构建一个完整的推理路径。

**函数原型**

```python
def sample_path(self, query, task_desc, max_path_length)
```

**参数说明**

| 参数名               | 数据类型 | 可选/必选 | 默认值 | 说明                   |
| ----------------- | ---- | ----- | --- | -------------------- |
| query             | str  | 必选    | -   | 原始查询文本。              |
| task\_desc        | str  | 必选    | -   | 任务描述，指导LLM的行为。       |
| max\_path\_length | int  | 可选    | 3   | 最大路径长度，表示生成子查询的最大轮数。 |

**返回值说明**

| 数据类型          | 说明                        |
| ------------- | ------------------------- |
| ReasoningPath | 包含完整推理路径的ReasoningPath对象。 |

#### generate\_final\_answer

**功能描述**

基于生成的推理路径，整合所有信息生成最终答案。

**函数原型**

```python
def generate_final_answer(self, rag_path, task_description)
```

**参数说明**

| 参数名                  | 数据类型          | 可选/必选 | 默认值  | 说明                      |
| -------------------- | ------------- | ----- | ---- | ----------------------- |
| rag\_path            | ReasoningPath | 必选    | -    | 包含推理路径的ReasoningPath对象。 |
| task\_description    | str           | 必选    | -    | 任务描述，指导LLM的行为。          |

**返回值说明**

| 数据类型 | 说明         |
| ---- | ---------- |
| str  | 生成的最终答案文本。 |

### SampleGenerator

#### 类功能

**功能描述**

样本生成器类，负责生成CoRAG训练样本，通过多线程并行处理输入数据，为每个查询生成有效的推理路径，并将其转换为可用于训练的样本格式。

**函数原型**

```python
from mx_rag.corag.sample_generator import SampleGenerator
SampleGenerator(config)
```

**参数说明**

| 参数名    | 数据类型            | 可选/必选 | 说明                                                                |
| ------ | --------------- | ----- | ----------------------------------------------------------------- |
| config | CoRagBaseConfig | 必选    | 配置对象，包含LLM实例、API地址和并行参数等。具体可参见[CoRagBaseConfig](#coragbaseconfig) |

#### 调用示例

```python
from mx_rag.corag.sample_generator import SampleGenerator
from mx_rag.corag.config import CoRagBaseConfig
from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils import ClientParam

# 初始化LLM实例
llm = Text2TextLLM(base_url="https://{ip}:{port}/v1/chat/completions",
                   model_name="qianwen-7b",
                   llm_config=LLMParameterConfig(max_tokens=512),
                   client_param=ClientParam(ca_file="/path/to/ca.crt")
                   )


# 初始化配置
config = CoRagBaseConfig(
    base_llm=llm,
    retrieve_api_url="http://your-retrieve-api.com/query",
    num_threads=4,
    max_path_length=3,
    client_param=ClientParam(ca_file="/path/to/ca.crt")
)

# 初始化样本生成器
generator = SampleGenerator(config)

# 生成训练样本
samples = generator.generate(
    input_file="data/train_queries.json",
    output_file="results/corag_train_samples.jsonl",
    n_samples=3
)

print("生成的样本数量:", sum(len(query_samples) for query_samples in samples))
```

#### generate

**功能描述**

生成样本主方法，从输入文件加载数据，并行处理生成训练样本，并保存到输出文件。

**函数原型**

```python
def generate(self, input_file, output_file, n_samples)
```

**参数说明**

| 参数名          | 数据类型 | 可选/必选 | 默认值 | 说明                                                                            |
| ------------ | ---- | ----- | --- | ----------------------------------------------------------------------------- |
| input\_file  | str  | 必选    | -   | 输入数据文件路径（JSONL格式），每条数据包含一个查询-答案对，示例：`{"query": "中国的首都是哪里?", "answer": "北京"}`。 |
| output\_file | str  | 必选    | -   | 输出文件路径。                                                                       |
| n\_samples   | int  | 可选    | 5   | 每个查询采样的路径数量。                                                                  |

**返回值说明**

| 数据类型                          | 说明                                                                                                                                                  |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| List\[List\[Dict\[str, Any]]] | 处理后的样本列表，示例：`{"type": "subquery_generation", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "SubQuery: ..."}]}`。 |

### 微调

#### FineTuneArguments

##### 类功能

**功能描述**

模型微调参数类，用于配置模型微调相关参数，包括模型路径、训练数据文件路径和最大序列长度等。

**函数原型**

```python
from mx_rag.corag import FineTuneArguments
FineTuneArguments(model_name_or_path, train_file, max_len)
```

**输入参数说明**

| 参数名                   | 数据类型           | 可选/必选 | 说明                                                 |
| --------------------- | -------------- | ----- | -------------------------------------------------- |
| model\_name\_or\_path | str            | 可选    | 预训练模型路径，仅支持本地模型，默认值为"Qwen/Qwen2.5-7B-Instruct"。    |
| train\_file           | Optional\[str] | 可选    | 训练数据文件路径（jsonl格式），默认值为"data/aligned\_train.jsonl"。 |
| max\_len              | int            | 可选    | 分词后的最大输入序列长度，默认值为2048。                             |

#### SubqueryFineTuner

##### 类功能

**功能描述**

子查询微调器类，用于微调模型以优化子查询生成。支持NPU加速，使用前需要调用`torch.npu.set_device`设置NPU设备。

**函数原型**

```python
from mx_rag.corag import SubqueryFineTuner
SubqueryFineTuner(finetune_args, train_args)
```

**输入参数说明**

| 参数名            | 数据类型              | 可选/必选 | 说明                                      |
| -------------- | ----------------- | ----- | --------------------------------------- |
| finetune\_args | FineTuneArguments | 必选    | 模型微调参数。                                 |
| train\_args    | TrainingArguments | 必选    | 训练参数，来自transformers库的TrainingArguments。 |

**核心方法**

##### train

**功能描述**

训练模型，执行模型准备、数据准备、训练器初始化，然后执行训练并保存模型。

**函数原型**

```python
def train(self)
```

**返回值说明**

无返回值，训练完成后会保存模型和分词器到指定目录。

#### 使用示例

**基本使用示例**

```python
from mx_rag.corag import SubqueryFineTuner, FineTuneArguments
from transformers import TrainingArguments
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

# 设置NPU设备
torch.npu.set_device(0)

# 配置微调参数
finetune_args = FineTuneArguments(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    train_file="data/aligned_train.jsonl",
    max_len=2048
)

# 配置训练参数
train_args = TrainingArguments(
    output_dir="./output",
    do_train=True,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    logging_dir="./logs",
    learning_rate=1e-5,
    logging_steps=10,
    save_steps=500,
    remove_unused_columns=False
)

# 创建微调器实例
tuner = SubqueryFineTuner(finetune_args, train_args)

# 执行训练
tuner.train()
```

### CoRagEvaluator

#### 类功能

**功能描述**

CoRAG评估器类，通过多线程并行处理评估数据，计算检索召回率等指标，并生成详细的评估报告。

**函数原型**

```python
from mx_rag.corag.evaluator import CoRagEvaluator
CoRagEvaluator(config)
```

**参数说明**

| 参数名    | 数据类型            | 可选/必选 | 说明                                                                |
| ------ | --------------- | ----- | ----------------------------------------------------------------- |
| config | CoRagBaseConfig | 必选    | 配置对象，包含LLM实例、API地址和并行参数等。具体可参见[CoRagBaseConfig](#coragbaseconfig) |

#### 调用示例

```python
from mx_rag.corag.evaluator import CoRagEvaluator
from mx_rag.corag.config import CoRagBaseConfig
from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils import ClientParam

# 初始化LLM实例
llm = Text2TextLLM(base_url="https://{ip}:{port}/v1/chat/completions",
                   model_name="qianwen-7b",
                   llm_config=LLMParameterConfig(max_tokens=512),
                   client_param=ClientParam(ca_file="/path/to/ca.crt")
                   )


# 初始化配置
config = CoRagBaseConfig(
    base_llm=llm,
    retrieve_api_url="http://your-retrieve-api.com/retrieve",
    num_threads=4,
    max_path_length=3,
    client_param=ClientParam(ca_file="/path/to/ca.crt")
)

# 初始化评估器
evaluator = CoRagEvaluator(config)

# 执行评估
eval_results = evaluator.evaluate(
    eval_file="data/eval_data.json",
    save_file="results/corag_eval_results.json",
    calc_recall=True,
    enable_naive_retrieval=True
)

# 输出评估结果
print("评估结果汇总:", eval_results[0])
```

#### evaluate

**功能描述**

执行评估主方法，从评估文件加载数据，并行处理生成评估结果，并保存到输出文件。

**函数原型**

```python
def evaluate(self, eval_file, save_file, calc_recall, enable_naive_retrieval, num_contexts)
```

**参数说明**

| 参数名                      | 数据类型 | 可选/必选 | 默认值  | 说明                                              |
| ------------------------ | ---- | ----- | ---- | ----------------------------------------------- |
| eval\_file               | str  | 必选    | -    | 评估数据文件路径（JSON格式）。支持HotpotQA和MuSiQue两种格式，详见下方示例。 |
| save\_file               | str  | 必选    | -    | 结果保存文件路径。                                       |
| calc\_recall             | bool | 可选    | True | 是否计算召回率。                                        |
| enable\_naive\_retrieval | bool | 可选    | True | 是否启用朴素检索对比。朴素检索是指通过原始问题直接调用检索API检索相关文档，不依赖CoRAG流程。 |
| num\_contexts            | int  | 可选    | 10   | 检索上下文数量。                                        |

**HotpotQA格式**

```json
[
  {
    "question": "Which company acquired by Google was founded first?",
    "answer": "YouTube",
    "context": [
      ["Title1", ["sentence1", "sentence2"]],
      ["Title2", ["sentence3", "sentence4"]]
    ],
    "supporting_facts": [
      ["Title1", [0, 1]],
      ["Title2", [0]]
    ]
  }
]
```

**MuSiQue格式**

```json
[
  {
    "question": "Which company acquired by Google was founded first?",
    "answer": "YouTube",
    "paragraphs": [
      {
        "paragraph_text": "Google is a multinational technology company that specializes in Internet-related services and products.",
        "is_supporting": false
      },
      {
        "paragraph_text": "YouTube is an American online video sharing and social media platform owned by Google. It was founded on February 14, 2005.",
        "is_supporting": true
      },
      {
        "paragraph_text": "Google Maps is a web mapping platform and consumer application offered by Google. It was first launched in February 2005.",
        "is_supporting": false
      }
    ]
  }
]
```

**返回值说明**

| 数据类型                   | 说明                                        |
| ---------------------- | ----------------------------------------- |
| List\[Dict\[str, Any]] | 评估结果列表，第一个元素是聚合指标，后续元素是每个样本的详细评估结果。详见下方示例 |

**评估输出**

```json
[
    {
        "type": "Summary",
        "total_samples": 11,
        "corag_accuracy": 0.36,
        "naive_accuracy": 0.090,
        "corag_correct_count": 4,
        "naive_correct_count": 1,
        "avg_path_time": 142.308,
        "avg_time": 56.682,
        "corag_micro_recall": 0.863,
        "naive_micro_recall": 0.68
    },
    {
        "question": "Who is the child of the performer of song Me And Bobby Mcgee?",
        "ground_truth": "Dean Miller",
        "corag_prediction": "xxx",
        "naive_prediction": "xxx",
        "is_correct": true,
        "naive_is_correct": false,
        "reasoning_steps": [
            {
                "subquery": "subquery1",
                "subanswer": "subanswer1"
            }
        ],
        "time": [
            144.0536253452301,
            66.77552223205566
        ],
        "corag_recall": {
            "hits": 1,
            "total": 2,
            "recall": 0.5
        },
        "naive_recall": {
            "hits": 1,
            "total": 2,
            "recall": 0.5
        }
    }, ...
]
```
