# API Reference - LLM Chain

# LLM Chain

The `Chain` definition implements integration with LLM clients.

## `Chain` Abstract Class

### Class Description

**Description**

The abstract base class for LLM chains. It defines the abstract interface.

**Prototype**

```python
from mx_rag.chain.base import Chain
Chain()
```

### `query`

**Description**

Use the `query` method to call the LLM for question answering.

**Prototype**

```python
def query(text, llm_config, *args, **kwargs)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|text|str|Required|Text to query.|
|llm_config|LLMParameterConfig|Optional|Parameters for calling the LLM. See [LLMParameterConfig](./llm_client.md#llmparameterconfig).|
|args|-|Optional|The valid arguments depend on the specific Chain.|
|kwargs|-|Optional|The valid arguments depend on the specific Chain.|

## `Text2ImgChain`

### Class Description

**Description**

Build a text-to-image LLM integration object that inherits from the abstract Chain class.

**Prototype**

```python
from mx_rag.chain import Text2ImgChain
Text2ImgChain(multi_model)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|multi_model|Text2ImgMultiModel|Required|LLM integration object. Pass an instance of [Text2ImgMultiModel](./llm_client.md#text2imgmultimodel).|

### `query`

**Description**

Generate an image from the given text prompt.

**Prototype**

```python
def query(text, llm_config, *args, **kwargs)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|text|str|Required|The image generation prompt. It must be non-empty, and the maximum length is 1024 * 1024.|
|llm_config|LLMParameterConfig|Optional|Parameters for calling the LLM. See [LLMParameterConfig](./llm_client.md#llmparameterconfig) for details.|
|args|list|Optional|Inherited from the base class and not used.|
|kwargs["output_format"]|str|Optional|The output image format, obtained from `kwargs["output_format"]`. Supported values are `["png", "jpeg", "jpg", "webp"]`. The default value is `png`.|
|kwargs["size"]|str|Optional|The image generation size, expressed as `height*width` and passed in through `kwargs`. The supported sizes depend on the corresponding LLM. The regular expression pattern is `"^\d{1,5}\\\*\d{1,5}$"`. The default value is `512 * 512`.|

**Return Values**

|Data Type|Description|
|--|--|
|Dict,{<br>"prompt": prompt, "result": data}|Where `data` is the image data encoded in base64.|

**Examples**

```python
from mx_rag.chain import Text2ImgChain
from mx_rag.llm import Text2ImgMultiModel
from mx_rag.utils import ClientParam
client_param = ClientParam(ca_file="/path/to/ca.crt")
multi_model=Text2ImgMultiModel(model_name="sd", url="text to img url", client_param=client_param)
text2img_chain = Text2ImgChain(multi_model=multi_model)
llm_data = text2img_chain.query("dog wearing black glasses", output_format="jpg")
print(llm_data)
```

## `Img2ImgChain`

### Class Description

**Description**

Build an image-to-image LLM integration object that inherits from Chain.

**Prototype**

```python
from mx_rag.chain import Img2ImgChain
Img2ImgChain(multi_model, retriever)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|multi_model|Img2ImgMultiModel|Required|LLM integration object. Pass an instance of [Img2ImgMultiModel](./llm_client.md#img2imgmultimodel).|
|retriever|BaseRetriever|Required|Similarity retriever. Pass an instance of [Retriever](./retrieval.md#retriever).|

### `query`

**Description**

Retrieve relevant images from the given text, combine them with the prompt, and send them to the LLM to generate an image.

**Prototype**

```python
def query(text, llm_config, *args, **kwargs)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|text|str|Required|Description text for image retrieval. The length range is `(0, 1 * 1000 * 1000]`.|
|llm_config|LLMParameterConfig|Optional|Inherited from the parent class method. Not used here.|
|args|list|Optional|Not used in the current chain.|
|kwargs["prompt"]|str|Required|The image generation prompt, passed in through `kwargs`. The length range is `(0, 1 * 1024 * 1024]`.|
|kwargs["size"]|str|Optional|The image generation size, expressed as `height*width` and passed in through `kwargs`. The supported sizes depend on the corresponding LLM. The regular expression pattern is `"^\d{1,5}\\\*\d{1,5}$"`. The default is `512*512`.|

**Return Values**

|Data Type|Description|
|--|--|
|Dict,{"prompt": prompt, "result": data}|Where `data` is the image data encoded in base64.|

**Examples**

```python
# This example retrieves relevant images from images uploaded to the knowledge base, combines them with the prompt, and sends them to the LLM to generate an image
from paddle.base import libpaddle
from mx_rag.chain import Img2ImgChain
from mx_rag.llm import Img2ImgMultiModel
from mx_rag.retrievers import Retriever
from mx_rag.storage.vectorstore import MindFAISS
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.embedding.local import ImageEmbedding
from mx_rag.utils import ClientParam
dev = 0
img_emb = ImageEmbedding(model_name="ViT-B-16", model_path="/path/to/chinese-clip-vit-base-patch16", dev_id=dev)
img_vector_store = MindFAISS(x_dim=512,
                             devs=[dev],
                             load_local_index="/path/to/image_faiss.index",
                             auto_save=True)
chunk_store = SQLiteDocstore(db_path="/path/to/sql.db")
img_retriever = Retriever(vector_store=img_vector_store, document_store=chunk_store,
                          embed_func=img_emb.embed_documents, k=1, score_threshold=0.5)
multi_model = Img2ImgMultiModel(model_name="sd",
                                url="img to image url",
                                client_param=ClientParam(ca_file="/path/to/ca.crt"))
img2img_chain = Img2ImgChain(multi_model=multi_model, retriever=img_retriever)
llm_data = img2img_chain.query("Find a picture of a little boy",
                               prompt="He is a knight, wearing armor, with a big sword in his right hand. Blur the background and focus on the knight.")
print(llm_data)
```

## `SingleText2TextChain`

### Class Description

**Description**

A single-turn dialogue chain. It implements basic question-answering functionality, inherits from Chain, and can also support multimodal dialogue. See [Basic dialogue function](#section175571825169) and [Multimodal dialogue function](#section175571825169).

**Prototype**

```python
from mx_rag.chain import SingleText2TextChain
SingleText2TextChain(llm, retriever, reranker, prompt, sys_messages, source, user_content_builder)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|llm|Text2TextLLM|Required|LLM object. See [Text2TextLLM](./llm_client.md#text2textllm).|
|retriever|Retriever|Required|Retriever object. See [Retriever](./retrieval.md#retriever).|
|reranker|Reranker|Optional|Reranker object used to rerank retrieved documents. The default is `None`. See [Reranker](./reranker.md#reranker).|
|prompt|str|Optional|You can add a system prompt while adding knowledge retrieval content to control the LLM more precisely. The default value is: "Based on the preceding known information, answer the user's question concisely and professionally. If the answer cannot be derived from the known information, answer based on your own experience." If you need a custom prompt, add it according to the LLM prompt engineering guidance. Length range: `[1, 1024 * 1024]`.|
|sys_messages|List[dict]|Optional|System messages. The default is `None`. The list can contain at most 16 items. Each dictionary can contain at most 16 key-value pairs. Each dictionary key string can be at most 16 characters. Each value string can be at most `4 * 1024 * 1024` characters. Example: `[{"role": "system", "content": "You are a friendly assistant"}]`.|
|source|bool|Optional|Whether to return the related documents retrieved during the conversation. The `source_documents` key in the Chain return dictionary is `True` by default.|
|user_content_builder|Callable|Optional|Callback function. The return value must be a string with a maximum length of `4 * 1024 * 1024`. The default function is `_user_content_builder`. Its purpose is to combine the three types of information, the original question, the retrieved document list, and the user prompt, and generate text that can directly serve as the `content` field of the `user` role message in the LLM conversation, that is, `{"role": "user", "content": generated result}`.|

- Default function for `user_content_builder`:

```python
def _user_content_builder(query: str, docs: List[Document], prompt: str) -> str:
    """
       Default logic for concatenating user input.
       Parameters:
       ----------
       query : str
           The user's original question.
           For example: "Please summarize the key points from the following material."
       docs : List[Document]
           The list of document objects returned by the retriever.
           Each `Document` usually contains:
           - page_content: the document text.
           - metadata: metadata such as source, title, and score.
       prompt : str
           The system prompt. The default is "Based on the preceding known information, answer the user's question concisely and professionally.
           If the answer cannot be derived from the known information, answer based on your own experience."
       Returns:
       -----
       str: The concatenated full prompt text, used as the LLM input.
       """
    final_prompt = ""
    document_separator: str = "\n\n"
    if len(docs) != 0:
        if prompt != "":
            last_doc = docs[-1]
            last_doc.page_content = (last_doc.page_content
                                     + f"{document_separator}{prompt}")
            docs[-1] = last_doc
        final_prompt = document_separator.join(x.page_content for x in docs)
    if final_prompt != "":
        final_prompt += document_separator
    final_prompt += query
    return final_prompt
```

### `query`

**Description**

RAG SDK dialogue function.

**Prototype**

```python
def query(text, llm_config, *args, **kwargs)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|text|str|Required|Original question. The range is `(0, 1000*1000]`.|
|llm_config|LLMParameterConfig|Optional|Parameters for calling the LLM. The default values are `temperature=0.5` and `top_p=0.95`. See [LLMParameterConfig](./llm_client.md#llmparameterconfig) for the remaining parameter descriptions.|
|args|List|Optional|Inherited from the parent class method signature. Not used here.|
|kwargs|Dictionary|Optional|Inherited from the parent class method signature. Not used here.|

**Return Values**

|Data Type|Description|
|--|--|
|Union[Dict, Iterator[Dict]]|The LLM return result. In the Dict, with knowledge source: `{"query": query, "result": data, "source_documents": [{'metadata': xxx, 'page_content': xxx}]}`. Without knowledge source: `{"query": query, "result": data}`.|

**Examples**

```python
from mx_rag.chain import SingleText2TextChain
from mx_rag.llm import Text2TextLLM
from mx_rag.embedding.local import TextEmbedding
from mx_rag.storage.vectorstore import MindFAISS
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.retrievers import Retriever
from mx_rag.utils import ClientParam
dev = 0
emb = TextEmbedding("/path/to/bge-large-zh-v1.5", dev_id=dev)
client_param = ClientParam(ca_file="/path/to/ca.crt")
llm = Text2TextLLM(model_name="Meta-Llama-3-8B-Instruct",
                   base_url="https://x.x.x.x:port/v1/chat/completions",
                   client_param=client_param)
vector_store = MindFAISS(x_dim=1024,  devs=[dev],
                                 load_local_index="/path/to/faiss.index",
                                 auto_save=True)
chunk_store = SQLiteDocstore(db_path="/path/to/sql.db")
retriever = Retriever(vector_store=vector_store, document_store=chunk_store, embed_func=emb.embed_documents, k=1, score_threshold=0.6)
rag = SingleText2TextChain(retriever=r, llm=llm)
response = rag.query("What modules does the mxVision software architecture include?", LLMParameterConfig(max_tokens=1024, temperature=1.0, top_p=0.1))
print(response)

```

- **Basic dialogue function<a id="section175571825169"></a>**

```python
from paddle.base import libpaddle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mx_rag.chain import SingleText2TextChain
from mx_rag.document import LoaderMng
from mx_rag.document.loader import DocxLoader, PdfLoader, PowerPointLoader
from mx_rag.embedding.local import TextEmbedding
from mx_rag.embedding.service import TEIEmbedding
from mx_rag.knowledge import KnowledgeDB, KnowledgeStore
from mx_rag.knowledge.handler import upload_files
from mx_rag.knowledge.knowledge import KnowledgeStore
from mx_rag.llm import Text2TextLLM, Img2TextLLM, LLMParameterConfig
from mx_rag.retrievers import Retriever
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.storage.vectorstore import MindFAISS
from mx_rag.utils import ClientParam
from mx_rag.llm.llm_parameter import LLMParameterConfig

loader_mng = LoaderMng()
# Load document loaders. You can use either the ones provided by mxrag or the ones from langchain
loader_mng.register_loader(loader_class=PdfLoader, file_types=[".pdf"])
loader_mng.register_loader(loader_class=DocxLoader, file_types=[".docx"])
loader_mng.register_loader(loader_class=PowerPointLoader, file_types=[".pptx"])
# Load document splitters, using the ones from langchain
loader_mng.register_splitter(splitter_class=RecursiveCharacterTextSplitter,
                             file_types=[".pdf", ".docx", ".txt", ".md", ".xlsx", ".pptx"],
                             splitter_params={"chunk_size": 750,
                                              "chunk_overlap": 150,
                                              "keep_separator": False })

dev = 0
# Load the embedding model
emb = TextEmbedding("/path/to/bge-large-zh-v1.5", dev_id=dev)
# Initialize the vector database
vector_store = MindFAISS(x_dim=1024,  devs=[dev],
                                 load_local_index="/path/to/faiss.index",
                                 auto_save=True)
# Initialize the document chunk relation database
chunk_store = SQLiteDocstore(db_path="/path/to/sql.db")
# Initialize the knowledge management relation database
knowledge_store = KnowledgeStore(db_path="/path/to/sql.db")
# Add the knowledge base
knowledge_store.add_knowledge("test", "Default", "admin")
# Initialize knowledge base management
knowledge_db = KnowledgeDB(knowledge_store=knowledge_store,
                           chunk_store=chunk_store,
                           vector_store=vector_store,
                           knowledge_name="test",
                           white_paths=["/path/"],
                           user_id="Default"
                           )
# Upload documents to the knowledge base
upload_files(knowledge_db, ["/path/to/file1", "/path/to/file2"], loader_mng, emb.embed_documents, True)
client_param = ClientParam(ca_file="/path/to/ca.crt")
llm = Text2TextLLM(model_name="Meta-Llama-3-8B-Instruct",
                   base_url="https://x.x.x.x:port/v1/chat/completions",
                   client_param=client_param)
r = Retriever(vector_store=vector_store, document_store=chunk_store, embed_func=emb.embed_documents, k=1, score_threshold=0.6)
rag = SingleText2TextChain(retriever=r, llm=llm)
response = rag.query("What modules does the mxVision software architecture include?", LLMParameterConfig(max_tokens=1024, temperature=1.0, top_p=0.1))
print(response)

```

- **Multimodal dialogue function**

```python
from paddle.base import libpaddle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mx_rag.chain import SingleText2TextChain
from mx_rag.document import LoaderMng
from mx_rag.document.loader import DocxLoader, PdfLoader, PowerPointLoader
from mx_rag.embedding.local import TextEmbedding
from mx_rag.embedding.service import TEIEmbedding
from mx_rag.knowledge import KnowledgeDB, KnowledgeStore
from mx_rag.knowledge.handler import upload_files
from mx_rag.knowledge.knowledge import KnowledgeStore
from mx_rag.llm import Text2TextLLM, Img2TextLLM, LLMParameterConfig
from mx_rag.retrievers import Retriever
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.storage.vectorstore import MindFAISS
from mx_rag.utils import ClientParam
from mx_rag.llm.llm_parameter import LLMParameterConfig
from typing import List
from langchain_core.documents import Document

# Load the multimodal LLM used to parse images in documents
vlm = Img2TextLLM(base_url="https://x.x.x.x:port/openai/v1/chat/completions",
                   model_name="Qwen2.5-VL-7B-Instruct",
                   llm_config=LLMParameterConfig(max_tokens=512),
                   client_param=ClientParam(ca_file="/path/to/ca.crt")
                   )
loader_mng = LoaderMng()
# Document loaders. You can use either the ones provided by mxrag or the ones from langchain
loader_mng.register_loader(loader_class=PdfLoader, file_types=[".pdf"], loader_params={"vlm": vlm})
loader_mng.register_loader(loader_class=DocxLoader, file_types=[".docx"], loader_params={"vlm": vlm})
loader_mng.register_loader(loader_class=PowerPointLoader, file_types=[".pptx"], loader_params={"vlm": vlm})
# Load document splitters, using the ones from langchain
loader_mng.register_splitter(splitter_class=RecursiveCharacterTextSplitter,
                             file_types=[".pdf", ".docx", ".txt", ".md", ".xlsx", ".pptx"],
                             splitter_params={"chunk_size": 750,
                                              "chunk_overlap": 150,
                                              "keep_separator": False })

dev = 0
# Load the embedding model
emb = TextEmbedding("/path/to/bge-large-zh-v1.5", dev_id=dev)
client_param = ClientParam(ca_file="/path/to/ca.crt")
# Initialize the vector database
vector_store = MindFAISS(x_dim=1024,  devs=[dev],
                                 load_local_index="/path/to/faiss.index",
                                 auto_save=True)
# Initialize the document chunk relation database
chunk_store = SQLiteDocstore(db_path="/path/to/sql.db")
# Initialize the knowledge management relation database
knowledge_store = KnowledgeStore(db_path="/path/to/sql.db")
# Add the knowledge base
knowledge_store.add_knowledge("test", "Default", "admin")
# Initialize knowledge base management
knowledge_db = KnowledgeDB(knowledge_store=knowledge_store,
                           chunk_store=chunk_store,
                           vector_store=vector_store,
                           knowledge_name="test",
                           white_paths=["/path/"],
                           user_id="Default"
                           )
# Upload documents to the knowledge base
upload_files(knowledge_db, ["/path/to/file1", "/path/to/file2"], loader_mng, emb.embed_documents, True)
# Define the callback function to combine the question with the retrieved text and image descriptions, and generate the content field of the `user` role message in the LLM conversation
def user_content_builder(query: str, docs: List[Document], *args, **kwargs):
       """
       Parameters:
       ----------
       query : str
           The user's original question. For example: "Please summarize the key points from the following material."
       docs : List[Document]
           The list of document objects returned by the retriever.
           Each `Document` usually contains: `page_content` for the document text and `metadata` for metadata such as source, title, and score.
       Returns:
       -----
       str : The concatenated full prompt text, used as the LLM input.
       """
    text_docs = [doc for doc in docs if doc.metadata.get("type", "") == "text"]
    img_docs = [doc for doc in docs if doc.metadata.get("type", "") == "image"]
    user_message = []
    if len(text_docs) > 0:
        # 2. Add text quotes
        user_message.append(f"Text Quotes are:")
        for i, doc in enumerate(text_docs):
            user_message.append(f"\n[{i + 1}] {doc.page_content}")
    if len(img_docs) > 0:
        # 3. Add image quotes, vlm-text or ocr-text
        user_message.append("\nImage Quotes are:")
        for i, doc in enumerate(img_docs):
            user_message.append(f"\nimage{i + 1} is described as: {doc.page_content}")
    user_message.append("\n\n")
    # 4. Add the user question
    user_message.append(f"The user question is: {query}")
    return ''.join(user_message)

# System prompt
TEXT_INFER_PROMPT = '''
You are a helpful question-answering assistant. Your task is to generate an interleaved text and image response based on the provided questions and quotes. Here is how to refine your process:

1. **Evidence Selection**:
   - From both the text and image quotes, identify the ones that are truly relevant to answering the question. Focus on significance and direct relevance.
   - Each image quote is the description of the image.

2. **Answer Construction**:
   - Use Markdown to embed text and images in your response. Avoid obvious headings or divisions, and ensure that the response flows naturally and coherently.
   - Conclude with a direct and concise answer to the question in a simple and clear sentence.

3. **Quote Citation**:
   - Cite images using the format `![{conclusion}](image index)`. For the first image, use `![{conclusion}](image1)`. The `{conclusion}` should be a concise one-sentence summary of the image's content.
   - Ensure that image citations strictly follow the format `![{conclusion}](image index)`. Do not simply state "See image1", "image1 shows", "[image1]", or "image1".
   - Each image or text can only be cited once.

- Do not cite irrelevant quotes.
- Compose a detailed and articulate interleaved answer to the question.
- Ensure that your answer is logical, informative, and directly tied to the evidence provided by the quotes.
- If a quote contains both text and image, the answer must contain both text and image responses.
- If a quote contains only text, the answer must contain only a text response and no image.
- Answer in Chinese.
'''

client_param = ClientParam(ca_file="/path/to/ca.crt")
# LLM used for dialogue
llm = Text2TextLLM(model_name="Meta-Llama-3-8B-Instruct",
                   base_url="https://x.x.x.x:port/v1/chat/completions",
                   client_param=client_param)
sys_messages=[{"role": "system", "content": TEXT_INFER_PROMPT}]
r = Retriever(vector_store=vector_store, document_store=chunk_store, embed_func=emb.embed_documents, k=1, score_threshold=0.6)
rag = SingleText2TextChain(retriever=r, llm=llm, sys_messages=sys_messages, user_content_builder=user_content_builder)
response = rag.query("What modules does the mxVision software architecture include?", LLMParameterConfig(max_tokens=1024, temperature=1.0, top_p=0.1))
# The source_documents in the response may contain images. You can obtain the image base64 encoding from the `metadata` field in the dictionary
print(response)
```

## `ParallelText2TextChain`

### Class Description

**Description**

Supports retrieval-based parallel-inference text-to-text dialogue. It inherits from the SingleText2TextChain base class and reduces retrieval latency.

**Prototype**

```python
from mx_rag.chain import ParallelText2TextChain
class ParallelText2TextChain(SingleText2TextChain)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|llm|Text2TextLLM|Required|LLM object. See [Text2TextLLM](./llm_client.md#text2textllm).|
|retriever|Retriever|Required|Retriever object. See [Retriever](./retrieval.md#retriever).|
|reranker|Reranker|Optional|Reranker object used to rerank retrieved documents. The default is `None`. See [Reranker](./reranker.md#reranker).|
|prompt|str|Optional|You can add a system prompt while adding knowledge retrieval content to control the LLM more precisely. The default value is: "Based on the preceding known information, answer the user's question concisely and professionally. If the answer cannot be derived from the known information, answer based on your own experience." If you need a custom prompt, add it according to the LLM prompt engineering guidance. Length range: `[1, 1024 * 1024]`.|
|sys_messages|List[dict]|Optional|System messages. The default is `None`. The list can contain at most 16 items. Each dictionary can contain at most 16 key-value pairs. Each dictionary key string can be at most 16 characters. Each value string can be at most `4 * 1024 * 1024` characters. Example: `[{"role": "system", "content": "You are a friendly assistant"}]`.|
|source|bool|Optional|Whether to return the related documents retrieved during the conversation. The `source_documents` key in the Chain return dictionary is `True` by default.|
|user_content_builder|Callable|Optional|Callback function. The return value must be a string with a maximum length of `4*1024*1024`. The default function is `_user_content_builder`. Its purpose is to combine the three types of information, the original question, the retrieved document list, and the user prompt, and generate text that can directly serve as the `content` field of the `user` role message in the LLM conversation, that is, `{"role": "user", "content": generated result}`.|

- Default function for `user_content_builder`:

```python
def _user_content_builder(query: str, docs: List[Document], prompt: str) -> str:
    """
       Default logic for concatenating user input.
       Parameters:
       ----------
       query : str
           The user's original question.
           For example: "Please summarize the key points from the following material."
       docs : List[Document]
           The list of document objects returned by the retriever.
           Each `Document` usually contains:
           - page_content: the document text.
           - metadata: metadata such as source, title, and score.
       prompt : str
           The system prompt. The default is "Based on the preceding known information, answer the user's question concisely and professionally.
           If the answer cannot be derived from the known information, answer based on your own experience."
       Returns:
       -----
       str: The concatenated full prompt text, used as the LLM input.
       """
    final_prompt = ""
    document_separator: str = "\n\n"
    if len(docs) != 0:
        if prompt != "":
            last_doc = docs[-1]
            last_doc.page_content = (last_doc.page_content
                                     + f"{document_separator}{prompt}")
            docs[-1] = last_doc
        final_prompt = document_separator.join(x.page_content for x in docs)
    if final_prompt != "":
        final_prompt += document_separator
    final_prompt += query
    return final_prompt
```

### `query`

**Description**

RAG SDK dialogue function.

**Prototype**

```python
def query(text: str, llm_config, *args, **kwargs)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|text|str|Required|Original question. The range is `(0, 1000*1000]`.|
|llm_config|LLMParameterConfig|Optional|Parameters for calling the LLM. The default values are `temperature=0.5` and `top_p=0.95`. See [LLMParameterConfig](./llm_client.md#llmparameterconfig) for the remaining parameter descriptions.|
|args|List|Optional|Inherited from the parent class method signature. Not used here.|
|kwargs|Dictionary|Optional|Inherited from the parent class method signature. Not used here.|

**Return Values**

|Data Type|Description|
|--|--|
|Union[Dict, Iterator[Dict]]|Returns a dictionary or an iterator. When `stream` is set to `True`, it returns an iterator. Otherwise, it returns a dictionary. The Dict contains:<li>With knowledge source: `{"prompt": prompt, "result": data, "source_documents": [{'metadata': xxx, 'page_content': xxx}]}`</li><li>Without knowledge source: `{"prompt": prompt, "result": data}`</li>|

**Examples**

```python
from mx_rag.chain import ParallelText2TextChain
from mx_rag.llm import Text2TextLLM
from mx_rag.embedding.local import TextEmbedding
from mx_rag.storage.vectorstore import MindFAISS
from mx_rag.storage.document_store import SQLiteDocstore
from mx_rag.retrievers import Retriever
from mx_rag.utils import ClientParam
dev = 0
emb = TextEmbedding("/path/to/bge-large-zh-v1.5", dev_id=dev)
client_param = ClientParam(ca_file="/path/to/ca.crt")
llm = Text2TextLLM(model_name="Meta-Llama-3-8B-Instruct",
                   base_url="https://x.x.x.x:port/v1/chat/completions",
                   client_param=client_param)
vector_store = MindFAISS(x_dim=1024,  devs=[dev],
                                 load_local_index="/path/to/faiss.index",
                                 auto_save=True)
chunk_store = SQLiteDocstore(db_path="/path/to/sql.db")
retriever = Retriever(vector_store=vector_store, document_store=chunk_store, embed_func=emb.embed_documents, k=1, score_threshold=0.6)
parallel_chain = ParallelText2TextChain(llm=llm, retriever=retriever)
answer = parallel_chain.query(text="123456")
print(answer)
```

## `GraphRagText2TextChain`

### Class Description

**Description**

A knowledge graph chain that inherits from [SingleText2TextChain](#singletext2textchain). For usage examples, see [Usage Example](./knowledge_graph.md#section8509453104117).

**Prototype**

```python
from mx_rag.chain.single_text_to_text import GraphRagText2TextChain
GraphRagText2TextChain(llm, retriever, reranker)
```

**Input Parameters**

|Parameter|Data Type|Required|Description|
|--|--|--|--|
|llm|Text2TextLLM|Required|LLM object. See [Text2TextLLM](./llm_client.md#text2textllm).|
|retriever|GraphRetriever|Required|GraphRetriever object returned by the [as_retriever](./knowledge_graph.md#as_retriever) method of [GraphRAGPipeline](./knowledge_graph.md#graphragpipeline).|
|reranker|Reranker|Optional|Reranker object used to rerank retrieved documents. The default is `None`. See [Reranker](./reranker.md#reranker).|

See the parent class for the remaining parameters.

### `query`

**Description**

Use this interface to perform question answering over knowledge graph data.

**Prototype**

```python
def query(text, llm_config, *args, **kwargs)
```

**Input Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|text|str|Required|Original question. The range is `(0, 1000*1000]`.|
|llm_config|LLMParameterConfig|Optional|Parameters for calling the LLM. The default values are `temperature=0.5` and `top_p=0.95`. See [LLMParameterConfig](./llm_client.md#llmparameterconfig) for the remaining parameter descriptions.|
|args|List|Optional|Inherited from the parent class method signature. Not used here.|
|kwargs|Dictionary|Optional|Inherited from the parent class method signature. Not used here.|

**Return Values**

|Data Type|Description|
|--|--|
|dict|Returns the LLM answer in the format `{'query': "who are Teutberga's parents?", 'result': "Teutberga's parents are Bosonid Boso the Elder and an unknown mother."}`.|
