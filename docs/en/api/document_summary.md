# Document Summary

## `Summary`

### Class Functionality

**Description**

This class extracts summaries from documents.

**Function Prototype**

```python
from mx_rag.summary import Summary
Summary(llm, llm_config)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|llm|Text2TextLLM|Required|An instance of the LLM object. For details about the specific type, see [Text2TextLLM](./llm_client.md#text2textllm).|
|llm_config|LLMParameterConfig|Optional|Parameters used to call the LLM. The default values are `temperature` set to `0.5` and `top_p` set to `0.95`. For descriptions of the remaining parameters, see [LLMParameterConfig](./llm_client.md#llmparameterconfig).|

**Usage Example**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mx_rag.document.loader import DocxLoader
from mx_rag.llm import Text2TextLLM
from mx_rag.summary import Summary
from mx_rag.utils import ClientParam
client_param = ClientParam(ca_file="/path/to/ca.crt")
llm = Text2TextLLM(base_url="https://ip:port/v1/chat/completions", model_name="qianwen-7b", client_param=client_param)
loader=DocxLoader("/workspace/MindIE.docx")
docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=150))
summary = Summary(llm=llm)
# Call the summarize method
sub_summaries = summary.summarize([doc.page_content for doc in docs])
# Call the merge_text_summarize method
res = summary.merge_text_summarize(sub_summaries)
print(res)
```

### `summarize`

**Description**

Uses an LLM to extract summaries from documents.

**Function Prototype**

```python
def summarize(texts, not_summarize_threshold, prompt)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|texts|List[str]|Required|List of input texts. The total length of all texts in the list ranges from `(0, 1024 * 1024]`. The list length ranges from `(0, 1024]`.|
|not_summarize_threshold|int|Optional|If a single summary request uses text that is too short, the LLM cannot summarize it or may produce an incorrect summary. This value sets the text length threshold above which the model must summarize the text. If the given text content is less than or equal to `not_summarize_threshold`, the model is not called to summarize it, and the original text is returned as the summary. The default value is `30`. The value range is `(0, 1024 * 1024]`.|
|prompt|`langchain_core.prompts.PromptTemplate`|Optional|The default value is as follows. The `input_variables` in `prompt` must equal `["text"]`, which indicates the input text. The `template` length ranges from `(0, 1024 * 1024]`. The actual query sent to the LLM is the concatenation of `prompt` and `text`. Its valid value depends on the MindIE configuration. For details, see the description of `maxSeqLen` in the "Core Concepts and Configuration > Configuration Parameter Description (Service-based)" section of the MindIE LLM Development Guide. Note that the language of `prompt` and `text` should ideally match, or you should specify the response language of the LLM. Otherwise, the model response quality may be affected.<br>`_SUMMARY_TEMPLATE = PromptTemplate(input_variables=["text"],`<br>`template="""Use concise language to extract a summary of the following content. Include as much key information as possible. Output only the content information. Answer in Chinese.\n\n{text}""")`|

**Return Values**

|Data Type|Description|
|--|--|
|List[str]|List of summarized texts.|

### `merge_text_summarize`

**Description**

Because of the LLM input token limit, long text must be split into multiple shorter texts for summarization. The method then summarizes the shorter texts to produce sub-summaries, merges the sub-summaries, and uses the LLM to summarize them again. After up to 10 iterations, it produces the final summary.

**Function Prototype**

```python
def merge_text_summarize(texts, merge_threshold, not_summarize_threshold, prompt)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|texts|List[str]|Required|List of text sub-summaries. The total length of all texts in the list ranges from `(0, 1024 * 1024]`. The list length ranges from `(0, 1024]`.|
|merge_threshold|int|Optional|When merging summaries, the LLM token limit requires you to split the sub-summary list and send the parts to the model for merged summarization. This value sets the split threshold and ensures that the total length of each split does not exceed the threshold. The default value is `4 * 1024`. The value range is `[1024, 1024 * 1024]`. The value of `merge_threshold` must be greater than the value of `not_summarize_threshold`.|
|not_summarize_threshold|int|Optional|If a single summary request uses text that is too short, the LLM cannot summarize it or may produce an incorrect summary. This value sets the text length threshold above which the model must summarize the text. If the given text content is less than or equal to `not_summarize_threshold`, the model is not called to summarize it, and the original text is returned as the summary. The default value is `30`. The value range is `(0, 1024 * 1024]`.|
|prompt|langchain_core.prompts.PromptTemplate|Optional|The default value is as follows. The `input_variables` in `prompt` must equal `["text"]`. The `template` length ranges from `(0, 1024 * 1024]`. The actual query sent to the LLM is the concatenation of `prompt` and `text`. Its valid value depends on the MindIE configuration. For details, see the description of `maxSeqLen` in the "Core Concepts and Configuration > Configuration Parameter Description (Service-based)" section of the MindIE LLM Development Guide.<br>Note that the language of `prompt` and `text` should ideally match, or you should specify the response language of the LLM. Otherwise, the model response quality may be affected.<br>`PromptTemplate(`<br>`input_variables=["text"],`<br>`template="""Use concise language to refine and merge the following multiple summaries into a single summary. Include as much key information as possible. Output only the content information. Answer in Chinese.\n\n{text}""")`|

**Return Values**

|Data Type|Description|
|--|--|
|str|Returns the final summary content after merging and summarization.|
