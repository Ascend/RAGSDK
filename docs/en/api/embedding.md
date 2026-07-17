# Embedding

## `TextEmbedding`

### Class Description<a name="ZH-CN_TOPIC_0000002452701717"></a>

**Description**

Starts the model locally with Transformers and provide text-to-vector embedding functionality. This class requires the weights of a BertModel-compatible model supported by Transformers. It inherits from the `langchain_core.embeddings.Embeddings` interface. The currently supported models are [BAAI/bge-large-zh-v1.5](https://www.modelscope.cn/models/BAAI/bge-large-zh-v1.5) and [aspire/acge_text_embedding](https://www.modelscope.cn/models/yangjhchs/acge_text_embedding).

> [!NOTE]
> If the configured model is not in the safetensors format, convert the model weights to the safetensors format before use. This prevents security risks that can arise from unsafe model weight formats such as ckpt and bin.

**Prototype**

```python
from mx_rag.embedding.local import TextEmbedding
TextEmbedding(model_path, dev_id, use_fp16, pooling_method, lock)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|model_path|str|Required|Directory of the model weight files. The path length cannot exceed 1024 characters. It cannot be a symbolic link or a relative path. Each file in the directory must be no larger than 10 GB, the directory depth cannot exceed 64, and the total number of files cannot exceed 512. The group of the running user, as well as users other than the running user, must not have write permission for files in the directory. The files in the directory and the group of the parent directory for those files must belong to the running user. The storage path cannot be in the following path list: [`/etc`, `/usr/bin`, `/usr/lib`, `/usr/lib64`, `/sys/`, `/dev/`, `/sbin`, `/tmp`].|
|dev_id|int|Optional|The NPU ID where the model runs. Run the **npu-smi info** command to query available IDs. Valid values are `[0, 63]`. The default is card 0.|
|use_fp16|bool|Optional|Whether to convert the model to half precision. The default is `True`.|
|pooling_method|str|Optional|The method used to process `last_hidden_state`. Valid values are [`cls`, `mean`, `max`, `lasttoken`]. The default is `cls`.|
|lock|multiprocessing.synchronize.Lock or _thread.LockType|Optional|The local model does not support multithreaded or multiprocess calls. If you need to call this interface from multiple processes or threads, a lock is required. The default value is `None`. Available values:<li>`None`: no lock is used. The interface does not support concurrency in this case.</li><li>`multiprocessing.Lock()`: a process lock. The interface supports multi-process calls in this case.</li><li>`threading.Lock()`: a thread lock. The interface supports multi-threaded calls in this case.</li>|

**Example Without Inference Acceleration**

```python
from paddle.base import libpaddle
from mx_rag.embedding.local import TextEmbedding
# Same as embed = TextEmbedding("/path/to/model", 1)
embed = TextEmbedding.create(model_path="/path/to/model", dev_id=1)
print(embed.embed_documents(['abc', 'bcd']))
print(embed.embed_query('abc'))
```

**Example With Inference Acceleration**

```python
import os
from paddle.base import libpaddle
import torch_npu
# Adapt the embedding inference acceleration
from mx_rag.transformer_adapter.modeling_bert_adapter import enable_bert_speed
# Enable embedding inference acceleration. Setting it to "True" enables the feature, and setting it to "False" disables it
os.environ["ENABLE_BOOST"] = "True"
from mx_rag.embedding.local import TextEmbedding
device_id = 1
torch_npu.npu.set_device(f"npu:{device_id}")
# Same as embed = TextEmbedding("/path/to/model", 1)
embed = TextEmbedding.create(model_path="/path/to/model", dev_id=device_id )
print(embed.embed_documents(['abc', 'bcd']))
print(embed.embed_query('abc'))
```

**Multithreaded Invocation Example (the other embedding models can also use this example)**

```python
from paddle.base import libpaddle
import threading
from mx_rag.embedding.local import TextEmbedding
def infer(k, embed):
    print(f"thread_{k}")
    print(embed.embed_query('abc'))
    print(embed.embed_documents(['abc', 'bcd']))

if __name__ == '__main__':
    worker_nums=2
    threads = []
    embed = TextEmbedding.create(model_path='/path/to/model', dev_id=1, pooling_method='cls', lock=threading.Lock())
    for i in range(worker_nums):
        thread = threading.Thread(target=infer, args=(i,embed,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
```

### `Create`

**Description**

Creates and return a `TextEmbedding` object.

**Prototype**

```python
@staticmethod
def create(**kwargs)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|kwargs|dict|Required|Keyword arguments. Refer to the input parameters in [Class Description](#ZH-CN_TOPIC_0000002452701717). Required parameters must be passed. Otherwise, a `KeyError` is raised.|

**Returns**

|Data Type|Description|
|--|--|
|TextEmbedding|`TextEmbedding` object.|

### `embed_documents`

**Description**

Uses the model to convert the text provided by the user into vectors.

**Prototype**

```python
def embed_documents(texts, batch_size)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|texts|List[str]|Required|A text list. The list length must be in the range `[1, 1000 * 1000]`. The length of each string must be in the range `[1, 128 * 1024 * 1024]`.|
|batch_size|int|Optional|The batch size. The method groups `batch_size` texts for each embedding operation. The value range is `[1, 1024]`. The default value is 32. The configurable value depends on the device memory.|

**Returns**

|Data Type|Description|
|--|--|
|List[List[float]]|The vector array converted from `texts`. If `texts` is an array of length 4 and the embedding model output is a 1024-dimensional vector, the final output is an array with the shape of `(4, 1024)`.|

### `embed_query`

**Description**

Uses the model to convert the text provided by the user into a vector.

**Prototype**

```python
def embed_query(text)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|text|str|Required|The text to convert to a vector. The text length must be in the range `[1, 128 * 1024 * 1024]`.|

**Returns**

|Data Type|Description|
|--|--|
|List[float]|The vector converted from `text`. If the embedding model output is a 1024-dimensional vector, the final output is an array with the shape of `(1, 1024)`.|

## `SparseEmbedding`

### Class Description<a id="ZH-CN_TOPIC_0000002419102844"></a>

**Description**

Starts the model locally with `transformers` and provide sparse text-to-vector embedding functionality. This class requires the weights of a `BertModel`-compatible model supported by `transformers`. It inherits from the `langchain_core.embeddings.Embeddings` interface. The currently supported model is BAAI/bge-m3.

> [!NOTE]
> If the configured model is not in the safetensors format, convert the model weights to the safetensors format before use. This prevents security risks that can arise from unsafe model weight formats such as ckpt and bin.

**Prototype**

```python
from mx_rag.embedding.local import SparseEmbedding
SparseEmbedding(model_path, dev_id, use_fp16)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|model_path|str|Required|Directory of the model weight files. The path length cannot exceed 1024 characters. It cannot be a symbolic link or a relative path.<li>Each file in the directory must be no larger than 10 GB, the directory depth cannot exceed 64, and the total number of files cannot exceed 512.</li><li>The group of the running user, as well as users other than the running user, must not have write permission for files in the directory.</li><li>The files in the directory and the group of the parent directory for those files must belong to the running user. The storage path cannot be in the following path list: ["/etc", "/usr/bin", "/usr/lib", "/usr/lib64", "/sys/", "/dev/", "/sbin", "/tmp"].</li>|
|dev_id|int|Optional|The NPU ID where the model runs. Run the **npu-smi info** command to query available IDs. Valid values are `[0, 63]`. The default is card 0.|
|use_fp16|bool|Optional|Whether to convert the model to half precision. The default is `True`.|

**Usage Example**

```python
from paddle.base import libpaddle
from mx_rag.embedding.local import SparseEmbedding
# Same as embed = SparseEmbedding("/path/to/model", 1)
embed = SparseEmbedding.create(model_path="/path/to/model", dev_id=1)
print(embed.embed_documents(['abc', 'bcd']))
print(embed.embed_query('abc'))
```

### `Create`

**Description**

Creates and return a `SparseEmbedding` object.

**Prototype**

```python
@staticmethod
def create(**kwargs)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|kwargs|dict|Required|Keyword arguments. Refer to the input parameters in [Class Description](#ZH-CN_TOPIC_0000002419102844). Required parameters must be passed. Otherwise, a `KeyError` is raised.|

**Returns**

|Data Type|Description|
|--|--|
|SparseEmbedding|`SparseEmbedding` object.|

### `embed_documents`

**Description**

Uses the model to convert the text provided by the user into vectors.

**Prototype**

```python
def embed_documents(texts, batch_size)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|texts|List[str]|Required|A text list. The list length must be in the range `[1, 1000 * 1000]`. The length of each string must be in the range `[1, 128 * 1024 * 1024]`.|
|batch_size|int|Optional|The batch size. The method groups `batch_size` texts for each embedding operation. The value range is `[1, 1024]`. The default value is 32. The configurable value depends on the device memory.|

**Returns**

|Data Type|Description|
|--|--|
|List[Dict[int, float]]|The vector array converted from `texts`. If `texts` is an array of length 4, the embedding model output is a dictionary with `token_id` as the key and `token_weights` as the value. The final output is an array with length 4, and each element is a dictionary.|

### `embed_query`

**Description**

Uses the model to convert the text provided by the user into vectors.

**Prototype**

```python
def embed_query(text)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|text|str|Required|The text to convert to a vector. The text length must be in the range `[1, 128 * 1024 * 1024]`.|

**Returns**

|Data Type|Description|
|--|--|
|Dict[int, float]|The sparse vector converted from `text`.|

## `TEIEmbedding`

### Class Description<a id="ZH-CN_TOPIC_0000002452821613"></a>

**Description**

Connects to the TEI service and provide text-to-vector embedding functionality. This class inherits from the `langchain_core.embeddings.Embeddings` interface.

**Prototype**

```python
from mx_rag.embedding.service import TEIEmbedding
TEIEmbedding(url, client_param, embed_mode)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|url|str|Required|TEI embedding service address. The string length must be in the range `[1, 128]`. The service supports the `/v1/embed`, `/v1/embeddings`, and `/embed_sparse` interfaces.<br>The embedding service based on the TEI framework does not support the HTTPS protocol. For security, you can deploy an nginx service and place the service and the embedding service in a trusted network. When you use the service, the client accesses nginx over HTTPS, and nginx forwards requests to the embedding service.|
|client_param|ClientParam|Optional|HTTPS client configuration parameters. The default value is `ClientParam()`. For details, see [ClientParam](./universal_api.md#clientparam).|
|embed_mode|str|Optional|The vectorization type provided by the TEI service. The default is `dense`. The value can be only `sparse` or `dense`. `sparse` indicates sparse vectorization, and `dense` indicates dense vectorization. This parameter is deprecated.|

**Returns**

`TEIEmbedding` object.

**Usage Example**

```python
from paddle.base import libpaddle
from mx_rag.embedding.service import TEIEmbedding
from mx_rag.utils import ClientParam
# Same as tei_embed = TEIEmbedding("https://ip:port/embed", client_param=ClientParam(xxx))
tei_embed = TEIEmbedding.create(url="https://ip:port/embed",
                                client_param=ClientParam(ca_file="/path/to/ca.crt"))
print(tei_embed.embed_documents(['abc', 'bcd']))
print(tei_embed.embed_query('abc'))
```

### `Create`

**Description**

Creates and return a `TEIEmbedding` object.

**Prototype**

```python
@staticmethod
def create(**kwargs)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|kwargs|dict|Required|Keyword arguments. Refer to the input parameters in [Class Description](#ZH-CN_TOPIC_0000002452821613). Required parameters must be passed. Otherwise, a `KeyError` is raised.|

**Returns**

|Data Type|Description|
|--|--|
|TEIEmbedding|`TEIEmbedding` object.|

### `embed_documents`

**Description**

Calls the TEI service to convert the text list provided by the user into vectors.

**Prototype**

```python
def embed_documents(texts, batch_size)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|texts|List[str]|Required|A text list. The list length must be in the range `(0, 1000 * 1000]`. The length of each string must be in the range `[1, 128 * 1024 * 1024]`.|
|batch_size|int|Optional|The batch size. The method groups `batch_size` texts for each embedding operation. The value range is `[1, 1024]`. The default value is 32.|

**Returns**

|Data Type|Description|
|--|--|
|List[List[float]]|The vector array converted from `texts`. If `texts` is an array of length 4 and the embedding model output is a 1024-dimensional vector, the final output is an array with the shape of `(4, 1024)`.|

### `embed_query`

**Description**

Calls the TEI service to convert the text provided by the user into a vector.

**Prototype**

```python
def embed_query(text)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|text|str|Required|The text to convert to a vector. The length range is `[1, 128 * 1024 * 1024]`.|

**Returns**

|Data Type|Description|
|--|--|
|List[float]|The vector array converted from `text`. If the embedding model output is a 1024-dimensional vector, the final output is an array with the shape of `(1, 1024)`.|

## `CLIPEmbedding`

### Class Description<a id="ZH-CN_TOPIC_0000002419262704"></a>

**Description**

Connects to the CLIP service and provide text or image embedding functionality. This class inherits from the `langchain_core.embeddings.Embeddings` interface.

**Prototype**

```python
from mx_rag.embedding.service import CLIPEmbedding
CLIPEmbedding(url, client_param)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|url|str|Required|CLIP embedding service address. The length of the `url` string cannot exceed 128 characters.|
|client_param|ClientParam|Optional|HTTPS client configuration parameters. The default value is `ClientParam()`. For details, see [ClientParam](./universal_api.md#clientparam).|

**Returns**

`CLIPEmbedding` object.

**Usage Example**

```python
from paddle.base import libpaddle
from mx_rag.embedding.service import CLIPEmbedding
from mx_rag.utils import ClientParam
clip_embed = CLIPEmbedding.create(url="https://ip:port/encode",
                                  client_param=ClientParam(ca_file="/path/to/ca.crt"))
print(clip_embed.embed_documents(['abc', 'bcd']))
print(clip_embed.embed_query('abc'))
```

### `Create`

**Description**

Creates and return a `CLIPEmbedding` object.

**Prototype**

```python
@staticmethod
def create(**kwargs)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|kwargs|dict|Required|Keyword arguments. Refer to the input parameters in [Class Description](#ZH-CN_TOPIC_0000002419262704). Required parameters must be passed. Otherwise, a `KeyError` is raised.|

**Returns**

|Data Type|Description|
|--|--|
|CLIPEmbedding|`CLIPEmbedding` object.|

### `embed_documents`

**Description**

Calls the CLIP service to convert the text list provided by the user into vectors.

**Prototype**

```python
def embed_documents(texts, batch_size)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|texts|List[str]|Required|A text list. The list length must be in the range `(0, 1000 * 1000]`. The length of each string must be in the range `[1, 128 * 1024 * 1024]`.|
|batch_size|int|Optional|The batch size. The method groups `batch_size` texts for each embedding operation. The value range is `[1, 1024]`. The default value is 32.|

**Returns**

|Data Type|Description|
|--|--|
|List[List[float]]|The vector array converted from `texts`. If `texts` is an array of length 4 and the embedding model output is a 512-dimensional vector, the final output is an array with the shape of `(4, 512)`.|

### `embed_images`

**Description**

Calls the CLIP service to convert the image list provided by the user into vectors.

**Prototype**

```python
def embed_images(images, batch_size)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|images|List[str]|Required|A list of images. The list length must be in the range `[1, 1000]`, and the length of each string must be in the range `[1, 10 * 1024 * 1024]`. Each image is a base64-encoded string.|
|batch_size|int|Optional|The batch size. The method groups `batch_size` texts for each embedding operation. The value range is `[1, 1024]`. The default value is 32. If `batch_size` is too large, the server may return a 500 error. In this case, reduce `batch_size`.|

**Returns**

|Data Type|Description|
|--|--|
|List[List[float]]|The vector array converted from `images`. If `images` is an array of length 4 and the embedding model output is a 512-dimensional vector, the final output is a list with the shape of `(4, 512)`.|

### `embed_query`

**Description**
