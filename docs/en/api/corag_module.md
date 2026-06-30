# CoRAG Interface Reference

# CoRAG Module

CoRAG (Chain of Retrieval-Augmented Generation) is a multi-turn question-answering framework based on chained retrieval-augmented generation. It iteratively generates subqueries, retrieves relevant documents, and integrates the information to support deep reasoning and answering complex questions.

## `CoRagBaseConfig`

### Class Overview

**Function Description**

The CoRAG base configuration class contains shared core parameters used to initialize CoRAG-related components.

**Function Prototype**

```python
from mx_rag.corag.config import CoRagBaseConfig
CoRagBaseConfig(base_llm, retrieve_api_url, num_threads, max_path_length, final_llm, sub_answer_llm, judge_llm)
```

**Parameters**

| Parameter       | Data Type    | Optional/Required | Default Value    | Description                                                                                              |
| ------------------- | ------------ | ----------------- | ---------------- | -------------------------------------------------------------------------------------------------------- |
| base\_llm           | Text2TextLLM | Required          | -                | Base LLM instance used to generate subqueries and answers. See [Text2TextLLM](./llm_client.md#text2textllm) for details. |
| retrieve\_api\_url  | str          | Required          | -                | URL of the retrieval API, used to fetch relevant documents. It must support POST requests. The request body must be JSON and include the query text `query`. The response body must also be JSON and can support multiple structures. See the request and response body examples below. |
| num\_threads        | int          | Optional          | 8                | Number of threads used for parallel processing.                                                          |
| max\_path\_length   | int          | Optional          | 3                | Maximum path length, which indicates the maximum number of rounds used to generate subqueries.          |
| final\_llm          | Text2TextLLM | Optional          | None             | LLM instance used to generate the final answer. If you do not provide one, `base_llm` is used. See [Text2TextLLM](./llm_client.md#text2textllm) for details. |
| sub\_answer\_llm    | Text2TextLLM | Optional          | None             | LLM instance used to generate subanswers. If you do not provide one, `base_llm` is used. See [Text2TextLLM](./llm_client.md#text2textllm) for details. |
| judge\_llm          | Text2TextLLM | Optional          | None             | LLM instance used to evaluate answer correctness. See [Text2TextLLM](./llm_client.md#text2textllm) for details. |
| retrieve\_top\_k    | int          | Optional          | 5                | Number of documents returned by the retrieval API. The default value is 5.                               |
| client\_param       | ClientParam  | Optional          | ClientParam()  | HTTP client parameters used to configure request behavior. See [ClientParam](./universal_api.md#clientparam) for details. |

**Request Body Example**

```json
{
    "query": "Which company acquired by Google was founded first?", "top_k": 5
}
```

**Response Body Examples**

The API supports multiple response formats. Common examples are as follows.

### Format 1: Standard Format with `document_ids` and `documents`

```json
{
    "document_ids": ["doc1", "doc2"],
    "documents": ["Google is a multinational technology company.", "YouTube was founded on February 14, 2005."]
}
```

### Format 2: Format with the `chunks` Field

```json
{
    "chunks": [
        "Google is a multinational technology company that specializes in Internet-related services and products.",
        "YouTube is an American online video sharing and social media platform owned by Google."
    ]
}
```

### Format 3: Format with the `data` Field

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

### Format 4: Format with the `results` Field

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

### Format 5: Format with the `docs` Field

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

### Format 6: Format with the `passages` Field

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

**Supported Response Field Descriptions**

- Document content can be extracted from the following fields: `content`, `contents`, and `text`.
- Document IDs can be extracted from the following fields: `id` and `doc_id`.
- The API also supports returning a list of strings directly or a list of dictionaries containing the preceding fields.

## `ReasoningPath`

### Class Overview

**Function Description**

This data class represents a CoRAG reasoning path. It contains lists of the original query, subqueries, subanswers, document IDs, reasoning steps, and documents.

**Function Prototype**

```python
from mx_rag.corag.corag_agent import ReasoningPath
ReasoningPath(original_query, subqueries, subanswers, document_ids, reasoning_steps, documents)
```

**Parameters**

| Parameter       | Data Type            | Optional/Required | Default Value | Description                                              |
| ------------------- | ------------------- | ----------------- | ------------- | -------------------------------------------------------- |
| original\_query     | str                 | Required          | -             | Original query text.                                     |
| subqueries          | List\[str]          | Optional          | \[]           | List of subqueries.                                      |
| subanswers          | List\[str]          | Optional          | \[]           | List of subanswers.                                      |
| document\_ids       | List\[List\[str]]   | Optional          | \[]           | List of document IDs. Each subquery corresponds to multiple document IDs. |
| reasoning\_steps    | List\[str]          | Optional          | \[]           | List of reasoning steps. Each subquery corresponds to one reasoning step. |
| documents           | List\[List\[str]]   | Optional          | \[]           | List of document contents. Each subquery corresponds to multiple document contents. |

## `CoRagAgent`

### Class Overview

**Function Description**

The CoRAG agent class generates reasoning paths and final answers. It is the core component of the CoRAG framework.

**Function Prototype**

```python
from mx_rag.corag.corag_agent import CoRagAgent
CoRagAgent(base_llm, retrieve_api_url, final_llm, sub_answer_llm)
```

**Parameters**

| Parameter      | Data Type    | Optional/Required | Default Value | Description                                                                                              |
| ------------------ | ------------ | ----------------- | ------------- | -------------------------------------------------------------------------------------------------------- |
| base\_llm          | Text2TextLLM | Required          | -             | Base LLM instance used to generate subqueries and answers. See [Text2TextLLM](./llm_client.md#text2textllm) for details. |
| retrieve\_api\_url | str          | Required          | -             | URL of the retrieval API, used to fetch relevant documents.                                              |
| final\_llm         | Text2TextLLM | Optional          | None          | LLM instance used to generate the final answer. If you do not provide one, `base_llm` is used. See [Text2TextLLM](./llm_client.md#text2textllm) for details. |
| sub\_answer\_llm   | Text2TextLLM | Optional          | None          | LLM instance used to generate subanswers. If you do not provide one, `base_llm` is used. See [Text2TextLLM](./llm_client.md#text2textllm) for details. |
| retrieve\_top\_k   | int          | Optional          | 5             | Number of documents returned by the retrieval API. The default value is 5.                               |
| client\_param      | ClientParam  | Optional          | ClientParam() | HTTP client parameters used to configure request behavior. See [ClientParam](./universal_api.md#clientparam) for details. |

### Usage Example

```python
from mx_rag.corag.corag_agent import CoRagAgent
from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils import ClientParam

# Initialize the LLM instance
llm = Text2TextLLM(base_url="https://{ip}:{port}/v1/chat/completions",
                   model_name="qianwen-7b",
                   llm_config=LLMParameterConfig(max_tokens=512),
                   client_param=ClientParam(ca_file="/path/to/ca.crt")
                   )

# Initialize the CoRagAgent
agent = CoRagAgent(
    base_llm=llm,
    retrieve_api_url="http://your-retrieve-api.com/retrieve",
    retrieve_top_k=5,
    client_param=ClientParam(ca_file="/path/to/ca.crt")
)

# Generate the reasoning path
task_desc = "Answer the user's complex question by retrieving relevant information through multiple rounds of subqueries."
rag_path = agent.sample_path(
    query="What is the working principle of the CoRAG framework?",
    task_desc=task_desc,
    max_path_length=3
)

# Generate the final answer
final_answer = agent.generate_final_answer(
    rag_path=rag_path,
    task_description=task_desc
)

print("Final answer:", final_answer)
```

### `sample_path`

**Function Description**

Iteratively generates subqueries, retrieves relevant documents from the data source based on the subqueries, and collects subanswers and relevant documents to build a complete reasoning path.

**Function Prototype**

```python
def sample_path(self, query, task_desc, max_path_length)
```

**Parameters**

| Parameter      | Data Type | Optional/Required | Default Value | Description                   |
| ------------------ | --------- | ----------------- | ------------- | ----------------------------- |
| query              | str       | Required          | -             | Original query text.          |
| task\_desc         | str       | Required          | -             | Task description that guides LLM behavior. |
| max\_path\_length  | int       | Optional          | 3             | Maximum path length, which indicates the maximum number of rounds used to generate subqueries. |

**Return Values**

| Data Type     | Description                                      |
| ------------- | ------------------------------------------------ |
| ReasoningPath | A `ReasoningPath` object that contains the complete reasoning path. |

### `generate_final_answer`

**Function Description**

Generates the final answer by integrating all information based on the generated reasoning path.

**Function Prototype**

```python
def generate_final_answer(self, rag_path, task_description)
```

**Parameters**

| Parameter       | Data Type      | Optional/Required | Default Value | Description                      |
| ------------------- | ------------- | ----------------- | ------------- | -------------------------------- |
| rag\_path           | ReasoningPath | Required          | -             | `ReasoningPath` object that contains the reasoning path. |
| task\_description   | str           | Required          | -             | Task description that guides LLM behavior. |

**Return Values**

| Data Type | Description                    |
| --------- | ------------------------------ |
| str       | Text of the generated final answer. |

## `SampleGenerator`

### Class Overview

**Function Description**

The sample generator class creates CoRAG training samples. It processes input data in parallel with multiple threads, generates a valid reasoning path for each query, and converts it into a sample format that can be used for training.

**Function Prototype**

```python
from mx_rag.corag.sample_generator import SampleGenerator
SampleGenerator(config)
```

**Parameters**

| Parameter | Data Type        | Optional/Required | Description                                                                 |
| -------------- | ---------------- | ----------------- | --------------------------------------------------------------------------- |
| config         | CoRagBaseConfig  | Required          | Configuration object that contains the LLM instance, API address, and parallel parameters. See [CoRagBaseConfig](#coragbaseconfig) for details. |

### Usage Example

```python
from mx_rag.corag.sample_generator import SampleGenerator
from mx_rag.corag.config import CoRagBaseConfig
from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils import ClientParam

# Initialize the LLM instance
llm = Text2TextLLM(base_url="https://{ip}:{port}/v1/chat/completions",
                   model_name="qianwen-7b",
                   llm_config=LLMParameterConfig(max_tokens=512),
                   client_param=ClientParam(ca_file="/path/to/ca.crt")
                   )


# Initialize the configuration
config = CoRagBaseConfig(
    base_llm=llm,
    retrieve_api_url="http://your-retrieve-api.com/query",
    num_threads=4,
    max_path_length=3,
    client_param=ClientParam(ca_file="/path/to/ca.crt")
)

# Initialize the sample generator
generator = SampleGenerator(config)

# Generate training samples
samples = generator.generate(
    input_file="data/train_queries.json",
    output_file="results/corag_train_samples.jsonl",
    n_samples=3
)

print("Number of generated samples:", sum(len(query_samples) for query_samples in samples))
```

### `generate`

**Function Description**

Main method for generating samples. It loads data from the input file, processes it in parallel to generate training samples, and saves the results to the output file.

**Function Prototype**

```python
def generate(self, input_file, output_file, n_samples)
```

**Parameters**

| Parameter | Data Type | Optional/Required | Default Value | Description                                                                                                  |
| -------------- | --------- | ----------------- | ------------- | ------------------------------------------------------------------------------------------------------------ |
| input\_file    | str       | Required          | -             | Path to the input data file in JSONL format. Each record contains a query-answer pair, for example: `{"query": "What is the capital of China?", "answer": "Beijing"}`. |
| output\_file   | str       | Required          | -             | Path to the output file.                                                                                    |
| n\_samples     | int       | Optional          | 5             | Number of paths sampled for each query.                                                                     |

**Return Values**

| Data Type                       | Description                                                                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| List\[List\[Dict\[str, Any]]]   | Processed sample list. Example: `{"type": "subquery_generation", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "SubQuery: ..."}]}`. |

## Fine Tuning

### `FineTuneArguments`

#### Class Overview

**Function Description**

The model fine-tuning parameter class configures parameters related to model fine-tuning, including the model path, training data file path, and maximum sequence length.

**Function Prototype**

```python
from mx_rag.corag import FineTuneArguments
FineTuneArguments(model_name_or_path, train_file, max_len)
```

**Input Parameters**

| Parameter          | Data Type       | Optional/Required | Description                                                                                 |
| ---------------------- | --------------- | ----------------- | ------------------------------------------------------------------------------------------- |
| model\_name\_or\_path  | str             | Optional          | Path to the pretrained model. Only local models are supported. The default value is "Qwen/Qwen2.5-7B-Instruct". |
| train\_file            | Optional\[str]  | Optional          | Path to the training data file in JSONL format. The default value is "data/aligned_train.jsonl". |
| max\_len               | int             | Optional          | Maximum input sequence length after tokenization. The default value is 2048.               |

### `SubqueryFineTuner`

#### Class Overview

**Function Description**

The subquery fine-tuner class fine-tunes the model to optimize subquery generation. It supports NPU acceleration. Before you use it, call `torch.npu.set_device` to set the NPU device.

**Function Prototype**

```python
from mx_rag.corag import SubqueryFineTuner
SubqueryFineTuner(finetune_args, train_args)
```

**Input Parameters**

| Parameter  | Data Type          | Optional/Required | Description                                                               |
| -------------- | ----------------- | ----------------- | ------------------------------------------------------------------------- |
| finetune\_args | FineTuneArguments | Required          | Fine-tuning parameters for the model.                                     |
| train\_args    | TrainingArguments | Required          | Training parameters from `TrainingArguments` in the transformers library. |

**Core Method**

#### `train`

**Function Description**

Trains the model. This method prepares the model, prepares the data, initializes the trainer, and then runs training and saves the model.

**Function Prototype**

```python
def train(self)
```

**Return Values**

No return value. After training finishes, the model and tokenizer are saved to the specified directory.

### Usage Example

**Basic Usage Example**

```python
from mx_rag.corag import SubqueryFineTuner, FineTuneArguments
from transformers import TrainingArguments
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

# Set the NPU device
torch.npu.set_device(0)

# Configure fine-tuning parameters
finetune_args = FineTuneArguments(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    train_file="data/aligned_train.jsonl",
    max_len=2048
)

# Configure training parameters
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

# Create the fine-tuner instance
tuner = SubqueryFineTuner(finetune_args, train_args)

# Run training
tuner.train()
```

## `CoRagEvaluator`

### Class Overview

**Function Description**

The CoRAG evaluator class processes evaluation data in parallel with multiple threads, computes metrics such as retrieval recall, and generates detailed evaluation reports.

**Function Prototype**

```python
from mx_rag.corag.evaluator import CoRagEvaluator
CoRagEvaluator(config)
```

**Parameters**

| Parameter | Data Type        | Optional/Required | Description                                                                 |
| -------------- | ---------------- | ----------------- | --------------------------------------------------------------------------- |
| config         | CoRagBaseConfig  | Required          | Configuration object that contains the LLM instance, API address, and parallel parameters. See [CoRagBaseConfig](#coragbaseconfig) for details. |

### Usage Example

```python
from mx_rag.corag.evaluator import CoRagEvaluator
from mx_rag.corag.config import CoRagBaseConfig
from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils import ClientParam

# Initialize the LLM instance
llm = Text2TextLLM(base_url="https://{ip}:{port}/v1/chat/completions",
                   model_name="qianwen-7b",
                   llm_config=LLMParameterConfig(max_tokens=512),
                   client_param=ClientParam(ca_file="/path/to/ca.crt")
                   )


# Initialize the configuration
config = CoRagBaseConfig(
    base_llm=llm,
    retrieve_api_url="http://your-retrieve-api.com/retrieve",
    num_threads=4,
    max_path_length=3,
    client_param=ClientParam(ca_file="/path/to/ca.crt")
)

# Initialize the evaluator
evaluator = CoRagEvaluator(config)

# Run evaluation
eval_results = evaluator.evaluate(
    eval_file="data/eval_data.json",
    save_file="results/corag_eval_results.json",
    calc_recall=True,
    enable_naive_retrieval=True
)

# Print the evaluation results
print("Evaluation summary:", eval_results[0])
```

### `evaluate`

**Function Description**

Main method for running evaluation. It loads data from the evaluation file, processes it in parallel to generate evaluation results, and saves the results to the output file.

**Function Prototype**

```python
def evaluate(self, eval_file, save_file, calc_recall, enable_naive_retrieval, num_contexts)
```

**Parameters**

| Parameter              | Data Type | Optional/Required | Default Value | Description                                                                                                        |
| -------------------------- | --------- | ----------------- | ------------- | ------------------------------------------------------------------------------------------------------------------ |
| eval\_file                 | str       | Required          | -             | Path to the evaluation data file in JSON format. It supports the HotpotQA and MuSiQue formats. See the examples below. |
| save\_file                 | str       | Required          | -             | Path used to save the results.                                                                                     |
| calc\_recall               | bool      | Optional          | True          | Whether to calculate recall.                                                                                      |
| enable\_naive\_retrieval   | bool      | Optional          | True          | Whether to enable a naive retrieval comparison. Naive retrieval means retrieving relevant documents by directly calling the retrieval API with the original question, without relying on the CoRAG workflow. |
| num\_contexts              | int       | Optional          | 10            | Number of retrieval contexts.                                                                                     |

**HotpotQA Format**

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

**MuSiQue Format**

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

**Return Values**

| Data Type                | Description                                                                 |
| ----------------------- | --------------------------------------------------------------------------- |
| List\[Dict\[str, Any]]  | List of evaluation results. The first element contains aggregated metrics, and the remaining elements contain detailed evaluation results for each sample. See the example below. |

**Evaluation Output**

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
