# API Reference

## Usage Notes

RAG SDK interfaces include knowledge management interfaces, database interfaces, model client interfaces, evaluation module interfaces, retrieval interfaces, and Chain interfaces. You can use these interfaces for custom development.

RAG SDK interfaces use an exception handling mechanism. Therefore, you must call them within `try/except` blocks and handle exceptions to prevent the program from exiting if an exception is raised during use.

> [!NOTICE]
>If RAG SDK uses Cache to cache question-answer pairs, the generated database is not encrypted at rest. Therefore, do not store personal data such as bank card numbers, ID card numbers, passport numbers, or passwords in the database.

|Interface Class|Link|
|--|--|
|General|[universal_api](./universal_api.md#general)|
|Knowledge Management|[knowledge_management](./knowledge_management.md#knowledge-management)|
|Databases|[databases](./databases.md#databases)|
|Model Client Integration|[llm_client](./llm_client.md)|
|Embedding|[embedding](./embedding.md)|
|Reranking|[reranker](./reranker.md)|
|Model Inference Acceleration|[model_inference_acceleration](./model_inference_acceleration.md)|
|Embedding Model Fine-Tuning|[embedding_model_fine_tuning](./embedding_model_fine_tuning.md)|
|Evaluation Module|[evaluation_module](./evaluation_module.md#evaluation-module)|
|Cache Module|[cache_module](./cache_module.md#cache-module)|
|Retrieval|[retrieval](./retrieval.md)|
|Document Summary|[document_summary](./document_summary.md#document-summary)|
|Prompt Compression|[prompt_compression](./prompt_compression.md)|
|LLM Chain|[llm_chains](./llm_chains.md)|
|Knowledge Graph|[knowledge_graph](./knowledge_graph.md)|
|CoRAG|[corag_module](./corag_module.md)|
