# Demo运行说明

## 功能描述

基于 RAG SDK 提供的向量模型垂域微调能力，自动完成训练数据生成、模型微调和效果评估的端到端流程。主要特性包括：

- **自动训练数据生成**：利用 LLM 从领域文档中自动生成问答对作为训练数据
- **迭代微调**：支持多轮迭代微调，每轮评估召回率提升，达到阈值自动终止
- **NPU 加速训练**：基于 `torch-npu` 在昇腾 NPU 上进行模型微调训练
- **效果评估**：使用 `InformationRetrievalEvaluator` 评估微调前后模型的召回率变化
- **模型合并**：微调后支持使用 LM-Cocktail 技术合并模型，缓解灾难性遗忘问题

## 前提条件

执行Demo前请先阅读[《RAG SDK 用户指南》](https://www.hiascend.com/document/detail/zh/mindsdk/730/rag/ragug/mxragug_0001.html)，并按照其中"安装部署"章节的要求完成必要软、硬件安装。
本章节为"应用开发"章节提供开发样例代码,便于开发者快速开发。

## 脚本执行

样例python脚本执行命令：

```bash
python3 finetune.py \
--document_path /home/embedding_finetune/rag_optimized/train_document \
--generate_dataset_path /home/embedding_finetune/rag_optimized/dataset \
--llm_url  http://51.38.68.109:1025/v1/chat/completions \
--llm_model_name Llama \
--use_http True \
--embedding_model_path /home/embedding_finetune/bge-large-zh-v1.5 \
--reranker_model_path /home/embedding_finetune/bge-reranker-large \
--finetune_output_path /home/embedding_finetune/rag_optimized/finetune_model \
--featured_percentage 0.8 \
--llm_threshold_score 0.8 \
--train_question_number 2 \
--query_rewrite_number 1 \
--eval_data_path /home/embedding_finetune/rag_optimized/eval/evaluate_data.jsonl \
--max_iter 3 \
--log_path /home/embedding_finetune/app.log \
--increase_rate 15
```

参数说明：

document_path：用于训练的原始文档路径，支持txt、md、doc格式

generate_dataset_path：数据集路径，生成的训练数据存放路径

llm_url：大模型推理接口地址

llm_model_name：接口地址对应的大模型名称

use_http：是否是http接口，默认False

embedding_model_path：embedding模型路径

reranker_model_path：reranker模型路径

finetune_output_path：微调模型的输出路径

featured_percentage：精选比例，bm25打分和reranker排序后保留的列表大小

llm_threshold_score：大模型打分优选分数阈值，只保留分数在阈值之上的QD对

train_question_number：针对切分的doc片段，每个doc片段产生的问题数

query_rewrite_number：query重写的次数

eval_data_path：评估数据路径，需要符合{"anchor": "query?", "positive": "answer."}这种格式，也可自定义key值，注意和代码对应

或者借助sdk辅助生成，生成后注意数据质量，需要手动过滤低质量数据

max_iter：最大迭代次数，对于切分后的doc数据来说，设定最大迭代次数，则每次取1/max_iter的数据（顺序取）参与训练数据生成

log_path：log文件保存路径

increase_rate：提升比例，当微调模型的召回率-原始模型的召回率超过了提升比例，则终止训练
