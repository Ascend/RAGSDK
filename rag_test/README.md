# RAG评测

RAG评测基于RAGAS，采用中文prompt，并对相应API和本地模型接口进行适配，使用户更易于使用，评测结果更准确。
包含以下评测指标：
| 评测指标  | 属性 | 说明 |
| ----- | -------- | -------- |
| faithfulness | 无参考度量指标 | 衡量了生成的答案与给定上下文的事实一致性 |
| answer_relevancy | 无参考度量指标 | 评估生成的答案与用户问题之间相关程度|
| context_relevancy | 无参考度量指标 | 衡量检索到的上下文的相关性 |
| context_recall | 基于真实答案度量指标 | 衡量检索到的上下文与人类提供的真实答案的一致程度 |
| context_precision | 基于真实答案度量指标 | 该指标确定最接近真实情况的上下文是否获得高分 |
| answer_correctness | 基于真实答案度量指标 | 生成答案与真实答案之间的准确性评估 |
| answer_similarity | 基于真实答案度量指标 | 对真实答案和生成答案之间的语义相似性进行评分 |
| critique | 基于真实答案度量指标 | 根据预定义的方面（如harmlessness和correctness）评估提交的内容 |
| context_entity_recall | 基于真实答案度量指标 | 确定真实答案中存在的所有实体是否也存在于所提供的上下文中|


## 基本使用方法
1. rag_test.py提供了rag测试入口，RAG_TEST_METRIC可以设置测试指标，RAG_TEST_LANGUAGE可以设置测试语言
2. 测试完成后在result文件夹下会生成相应的结果文件，格式为rag_test_currenttime.csv

## 评测模型选择
在rag_test.py中可以设置评测模型。

若选择API模型，则设置相应的url，以下为例：
embedding_model = APIEmbedding(emb_url)
llm_model = APILLM(llm_url)

若选择本地模型，则设置相应的模型路径，以下为例：
embedding_model = LocalEmbedding(llm_path)
llm_model = LocalLLM(embed_path)

## 评测数据集自动生成辅助工具
datatest文件夹下auto_datatest.py为评测数据集自动生成辅助工具，其主要功能为用户提供ground_truth的json文件，自动针对每个ground_truth生成多个问题，用户在此基础上筛选合适的问题作为相应的评测数据集。
