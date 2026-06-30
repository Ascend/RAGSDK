# API Reference - Prompt Compression

# Prompt Compression

## `PromptCompressor`

### Class Functionality

**Function description**

Abstract base class for prompt compression.

**Function prototype**

```python
from mx_rag.compress.base_compressor import PromptCompressor
class PromptCompressor(ABC)
```

### `compress_texts`

**Function description**

Compress prompt text.

**Function prototype**

```python
@abstractmethod
def compress_texts(self, context, question)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|context|str|Required|The long text to summarize.|
|question|str|Required|The instruction for summarizing the long text.|

## `RerankCompressor`

### Class Functionality

**Function description**

Use a ranking model to compute relevance scores between `question`, the instruction for summarizing the long text, and `context` chunks, which are slices of the long text to summarize. According to the configured compression rate threshold, preferentially retain the chunks with higher relevance, thereby enabling effective compression of long text.

**Function prototype**

```python
from mx_rag.compress.rerank_compressor import RerankCompressor
class RerankCompressor(reranker, splitter)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|reranker|Reranker|Required|The ranking model instance that performs fine-grained ranking on text chunks. It can only be the `Reranker` object from `mx_rag.reranker`. For details, see [Reranker](./reranker.md#reranker).|
|splitter|TextSplitter|Optional|The document splitting function. It can only be a subclass of LangChain `TextSplitter`. The default is `langchain.text_splitter.RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0, separators=["\n", ""], keep_separator=True)`.|

**Example**

```python
from mx_rag.compress.rerank_compressor import RerankCompressor
from mx_rag.reranker.local import LocalReranker
from mx_rag.reranker.service import TEIReranker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mx_rag.utils import ClientParam

context="""Prompt text to compress."""
question="Please give the preceding content a title."
tei_reranker=False
if tei_reranker:
    reranker = TEIReranker.create(url="https://ip:port/rerank",
                            client_param=ClientParam(ca_file="/path/to/ca.crt"))
else:
    reranker = LocalReranker(model_path="reranker_path", dev_id=0)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0, separators=["\n", ""], keep_separator=True)
compressor=RerankCompressor(reranker=reranker, splitter=text_splitter)
res=compressor.compress_texts(context, question, 0.3)
print(res)
```

### `compress_texts`

**Function description**

Compress text according to the instruction (`question`), the long text (`context`), and the compression rate (`compress_rate`).

**Function prototype**

```python
def compress_texts(context, question, compress_rate, context_reorder)
```

**Input parameter description**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|context|str|Required|The long text to summarize. Length range: [1, 16 MB].|
|question|str|Required|The instruction for summarizing the long text, which is used to calculate the relevance between it and the `context` text chunks. Length range: `[1, 1000 * 1000]`.|
|compress_rate|float|Optional|The compression rate. The default is 0.6. Valid range: `(0, 1)`.|
|context_reorder|bool|Optional|Whether to reorder based on scores. The default is `False`. If `True`, after relevance is computed, the text chunks with lower relevance are retained first according to the compression rate.|

**Returns**

|Data Type|Description|
|--|--|
|str|The compressed text.|

## `ClusterCompressor`

### Class Functionality

**Function description**

Use a clustering model to cluster embedded text and divide it into multiple semantic clusters. Then compute the cosine similarity between each `context` chunk and `question`, the instruction for summarizing the long text. According to the configured compression rate, remove the chunks with lower similarity within each cluster, thereby retaining the information most relevant to the instruction and achieving compressed summarization of long text.

**Function prototype**

```python
from mx_rag.compress.cluster_compressor import ClusterCompressor
class ClusterCompressor(cluster_func, embed, splitter, dev_id):
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|cluster_func|Callable[[List[List[float]]], Union[List[int], np.ndarray]]|Required|The clustering function. It clusters embedded text chunks into multiple semantic clusters. The return value must be `List[int]` or `ndarray`. The length cannot exceed `1000 * 1000`, and it must match the number of text chunks.|
|embed|Embeddings|Required|The embedding object that converts text chunks into vectors. It can only be a subclass that inherits from `langchain_core.embeddings.Embeddings`.|
|splitter|TextSplitter|Optional|The document splitting object. It can only be a subclass that inherits from `langchain_text_splitters.base.TextSplitter`. The default is `langchain.text_splitter.RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0, separators=["。", "！", "？", "\n", "，", "；", " ", ""])`.|
|dev_id|int|Optional|The NPU ID. Query available IDs with `npu-smi info`. Valid values are in the range `[0, 63]`. The default is card 0.|

**Example**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.cluster import HDBSCAN
from mx_rag.compress.cluster_compressor import ClusterCompressor
from mx_rag.embedding.local import TextEmbedding
from mx_rag.embedding.service import TEIEmbedding
from mx_rag.utils import ClientParam

context="""Prompt text to compress."""
question="Please give the preceding content a title."
tei_emb=False
if tei_emb:
    emb = TEIEmbedding.create(url="https://ip:port/embed", client_param=ClientParam(ca_file="/path/to/ca.crt"))
else:
    emb = TextEmbedding(model_path="embedding_path", dev_id=0)
def _get_community(sentences_embedding):
    # Cluster assignment
    node_num=len(sentences_embedding)
    min_cluster_size=2
    hdbscan = HDBSCAN(min_cluster_size=min(min_cluster_size, node_num))
    labels = hdbscan.fit_predict(sentences_embedding)
    return labels
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0, separators=["。", "！", "？", "\n", "，", "；", " ", ""], )
compressor=ClusterCompressor(cluster_func=_get_community, embed=emb, splitter=splitter, dev_id=0)
res=compressor.compress_texts(context, question, 0.6)
print(res)
```

### `compress_texts`

**Function description**

Compress text according to the provided instruction (`question`), long text (`context`), and compression rate (`compress_rate`).

**Function prototype**

```python
def compress_texts(context, question, compress_rate)
```

**Input parameter description**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|context|str|Required|The long text to summarize. Length range: [1, 16 MB].|
|question|str|Required|The instruction for summarizing the long text, which is used to calculate the relevance between it and the `context` text chunks. Length range: `[1, 1000 * 1000]`.|
|compress_rate|float|Optional|The compression rate. The default value is 0.6. Valid range: `(0, 1)`.|

**Returns**

|Data Type|Description|
|--|--|
|str|The compressed text.|
