# Model Client Integration

## `LLM`

### `Text2TextLLM`

#### Class Overview

**Description**

Connects a client to an LLM service and provides model interaction capabilities. Currently, it only supports the OpenAI-compatible `/v1/chat/completions` endpoint. This class inherits from `langchain.llms.base.LLM`.

**Prototype**

```python
from mx_rag.llm import Text2TextLLM
# All parameters must be passed as keyword arguments
Text2TextLLM(base_url, model_name, llm_config, client_param)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|base_url|str|Required|LLM service address. Length range: `[1, 128]`.|
|model_name|str|Required|LLM model name. Length range: `[1, 128]`.|
|llm_config|LLMParameterConfig|Optional|Takes effect when you call the model through LangChain. See [LLMParameterConfig](#llmparameterconfig) for details. When you call the model without LangChain, pass parameters through the `chat` and `chat_streamly` methods. See [chat](#chat) and [chat_streamly](#chat_streamly).|
|client_param|ClientParam|Optional|HTTPS client configuration parameters. The default value is `ClientParam()`. See [ClientParam](./universal_api.md#clientparam) for details.|

**Example**

```python
from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils import ClientParam
llm = Text2TextLLM(base_url="https://{ip}:{port}/v1/chat/completions",
                   model_name="qianwen-7b",
                   llm_config=LLMParameterConfig(max_tokens=512),
                   client_param=ClientParam(ca_file="/path/to/ca.crt")
                   )
res = llm.chat("Please introduce Beijing.")
print(res)
for res in llm.chat_streamly("Please introduce Beijing."):
    print(res)
```

#### `chat`

**Description**

Chat with the LLM service to obtain the inference result of the LLM model.

**Prototype**

```python
def chat(query, sys_messages, role, llm_config)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|query|str|Required|Inference request text. Length range: [1, 4 \* 1024 \* 1024].|
|sys_messages|List[dict]|Optional|System messages. The list can contain up to 16 items. Each dictionary can contain up to 16 items. Each dictionary key string can be up to 16 characters. Each value string can be up to 4 \* 1024 \* 1024 characters. The default value is `None`.|
|role|str|Optional|Role of the inference request message. Length range: `[1, 16]`. The default value is `user`.|
|llm_config|LLMParameterConfig|Optional|Parameters for calling the LLM. See [LLMParameterConfig](#llmparameterconfig) for details. The default value is `None`.|

**Returns**

|Data Type|Description|
|--|--|
|str|Result of LLM text inference.|

#### `chat_streamly`

**Description**

Chat with the LLM service to obtain the streaming inference result of the LLM model.

**Prototype**

```python
def chat_streamly(query, sys_messages, role, llm_config)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|query|str|Required|Inference request text. Length range: [1, 4 \* 1024 \* 1024].|
|sys_messages|List[dict]|Optional|System messages. The list can contain up to 16 items. Each dictionary can contain up to 16 items. Each dictionary key string can be up to 16 characters. Each value string can be up to 4 \* 1024 \* 1024 characters. The default value is `None`.|
|role|str|Optional|Role of the inference request message. Length range: `[1, 16]`. The default value is `user`.|
|llm_config|LLMParameterConfig|Optional|Parameters for calling the LLM. See [LLMParameterConfig](#llmparameterconfig) for details.|

**Returns**

|Data Type|Description|
|--|--|
|Iterator[str]|Streaming result of LLM text inference.|

## Image Generation Model

### `Text2ImgMultiModel`

#### Class Overview

**Description**

Connect to a text-to-image LLM service and provide model interaction capabilities.

Currently, only the following models are supported: `stable-diffusion-v1-5` and `stable-diffusion-2-1-base`.

**Prototype**

```python
from mx_rag.llm import Text2ImgMultiModel
Text2ImgMultiModel(url, model_name, client_param)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|url|str|Required|LLM access URL. Length range: `[1, 128]`.|
|model_name|str|Optional|SD model name. The default value is `None`. Length range: `(0, 128]`.|
|client_param|ClientParam|Optional|HTTPS client configuration parameters. The default value is `ClientParam()`. See [ClientParam](./universal_api.md#clientparam) for details.|

**Returns**

Text2ImgMultiModel object.

**Example**

```python
from mx_rag.llm import Text2ImgMultiModel
from mx_rag.utils import ClientParam
multi_model = Text2ImgMultiModel(model_name="sd", url="txt to image url",
                                 client_param=ClientParam(ca_file="/path/to/ca.crt"))
res = multi_model.text2img(prompt="dog wearing black glasses", output_format="jpg", size="512*512")
print(res)
```

#### `text2img`

**Description**

Interact with the SD service to generate images from text and obtain the inference result from the SD model.

The request body uses the following data format:

```python
{
"prompt": Text generation prompt,
"output_format": Output image format,
"size": Image generation size,
"model_name": Model name
}
```

**Prototype**

```python
def text2img(prompt, output_format, size)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|prompt|str|Required|Prompt used to generate the image. Length range: [1, 1024 \* 1024].|
|output_format|str|Optional|Format of the generated image. Supported values are `png`, `jpeg`, `jpg`, and `webp`. The default value is `png`.|
|size|str|Optional|Image generation size, expressed as `"height*width"`. The specific supported sizes depend on the corresponding LLM. The regular expression format is `^\d{1,5}\\\*\d{1,5}$`. The default value is `"512*512"`. The currently supported model generates images in `"512 * 512"`.|

**Returns**

|Data Type|Description|
|--|--|
|dict|Returned in the format `{"prompt": prompt, "result": data}`. `prompt` is the prompt used to generate the image, and `result` is the base64-encoded image data returned by the LLM inference result.|

### `Img2ImgMultiModel`

#### Class Overview

**Description**

Connect to an image-to-image LLM service and provide model interaction capabilities.

Currently, only the model built with IP-Adapter is supported: `stable-diffusion-v1-5`.

**Prototype**

```python
from mx_rag.llm import Img2ImgMultiModel
Img2ImgMultiModel(url, model_name, client_param)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|url|str|Required|LLM access URL. Length range: `[1, 128]`.|
|model_name|str|Optional|SD model name. The default value is `None`. Length range: `[1, 128]`.|
|client_param|ClientParam|Optional|HTTPS client configuration parameters. The default value is `ClientParam()`. See [ClientParam](./universal_api.md#clientparam) for details.|

**Returns**

Img2ImgMultiModel object.

**Example<a name="section175571825169"></a>**

```python

import sys
from mx_rag.document.loader import ImageLoader
from mx_rag.llm import Img2ImgMultiModel
from mx_rag.utils import ClientParam
multi_model = Img2ImgMultiModel(url="image to image url", model_name="sd",
                                client_param=ClientParam(ca_file="/path/to/ca.crt")
                                )
loader = ImageLoader("image path")
docs = loader.load()
if len(docs) < 1:
    print("load image failed")
    sys.exit(1)
res = multi_model.img2img(
    prompt="he is a knight, wearing armor, big sword in right hand. Blur the background, focus on the knight",
    image_content=docs[0].page_content,
    size="512*512")
print(res)

```

#### `img2img`

**Description**

Interact with the model service to perform image-to-image generation and obtain the model inference result. The request body format is as follows:

```python
Request body format:
{
"prompt": Prompt used to generate the image,
"image": Base64-encoded image data,
"size": Image generation size,
"model_name": Model name
}
```

**Prototype**

```python
def img2img(prompt, image_content, size)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|prompt|str|Required|Prompt used to generate the image. Length range: [1, 1024 \* 1024].|
|image_content|str|Required|String corresponding to the base64-encoded image data. Length range: (0, 10 \* 1024 \* 1024].|
|size|str|Optional|Image generation size, expressed as `"height*width"`. The specific supported sizes depend on the corresponding LLM. The regular expression format is `^\d{1,5}\\\*\d{1,5}$`. The default value is `"512 * 512"`.|

**Returns**

|Data Type|Description|
|--|--|
|dict|Returned in the format `{"prompt": prompt, "result": data}`. `prompt` is the prompt used to generate the image, and `result` is the base64-encoded image data returned by the LLM inference result.|

## Parameter Classes

### `LLMParameterConfig`

#### Class Overview

**Description**

Parameter class for connecting to LLMs. The specific valid value of each parameter varies by model configuration.

**Prototype**

```python
from mx_rag.llm import LLMParameterConfig
LLMParameterConfig(max_tokens, presence_penalty, frequency_penalty, temperature, top_p, seed, stream)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|max_tokens|int|Optional|Maximum number of tokens that inference can generate. Value range: `[1, 100000]`. The default value is 512. Pass this value through `kwargs`. The actual valid value depends on the MindIE configuration. For details, see the description of `maxSeqLen` in the "Core Concepts and Configuration > Configuration Parameter Description (Service-Based)" section of the MindIE LLM Development Guide.|
|presence_penalty|float, int|Optional|Controls how the model penalizes new tokens based on whether they have already appeared in the text so far. Positive values increase the likelihood that the model will talk about new topics by penalizing words it has already used. Value range: [-2.0, 2.0]. The default value is 0.0.|
|frequency_penalty|float, int|Optional|Controls how the model penalizes new tokens based on the existing frequency of words or tokens in the text. Positive values reduce the likelihood that the model repeats words on a line by penalizing words it has already used frequently. Value range: [-2.0, 2.0]. The default value is 0.0.|
|seed|int|Optional|Random seed used to specify the inference process. The same `seed` value ensures reproducible inference results, and different `seed` values increase the randomness of inference results. Value range: `[0, 2 ** 31 - 1]`. If you do not pass this parameter, the system generates a random `seed` value. The default value is `None`.|
|temperature|float, int|Optional|Controls generation randomness. Higher values produce more diverse output. Value range: `[0.0, 2.0]`. The default value is 1.0.|
|top_p|float, int|Optional|Controls the range of tokens considered during model generation. It uses cumulative probability to select candidate tokens until the cumulative probability exceeds the given threshold. This parameter also controls the diversity of generated results. It selects candidate tokens based on cumulative probability until the cumulative probability exceeds the given threshold. Value range: `(0.0, 1.0]`. The default value is 1.0.|
|stream|bool|Optional|Specifies whether to return a streaming response. The default value is `False`. This parameter takes effect in the following scenarios.<br>["ParallelText2TextChain", "SingleText2TextChain", "GraphRagText2TextChain"].|

**Example**

```python

from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils import ClientParam
llm = Text2TextLLM(base_url="https://{ip}:{port}/v1/chat/completions",
                   model_name="qianwen-7b",
                   llm_config=LLMParameterConfig(max_tokens=512),
                   client_param=ClientParam(ca_file="/path/to/ca.crt")
                   )
res = llm.chat("Please introduce Beijing.")
print(res)
for res in llm.chat_streamly("Please introduce Beijing."):
    print(res)
```

## Vision LLM

### `Img2TextLLM`

#### Class Overview

**Description**

This client connects to a vision LLM service and provides model interaction capabilities. Currently, it only supports the OpenAI-compatible `/openai/v1/chat/completions` endpoint. This class inherits from `langchain.llms.base.LLM`.

**Prototype**

```python
from mx_rag.llm import Img2TextLLM
# All parameters must be passed as keyword arguments
Img2TextLLM(base_url, prompt, model_name, llm_config, client_param)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|base_url|str|Required|LLM service address. Length range: `[1, 128]`.|
|prompt|str|Optional|Prompt text used to guide the vision LLM to generate structured, detailed, and well-formed image descriptions. The default value is [IMG_TO_TEXT_PROMPT](#li183183578215), and users can configure it as needed. Length range: `[1, 1024 * 1024]`.|
|model_name|str|Required|LLM model name. Length range: `[1, 128]`.|
|llm_config|LLMParameterConfig|Optional|Takes effect when you call the model through LangChain. See [LLMParameterConfig](#llmparameterconfig) for details. When you call the model without LangChain, pass parameters through the `chat` method. See [chat](#chat).|
|client_param|ClientParam|Optional|HTTPS client configuration parameters. The default value is `ClientParam()`. See [ClientParam](./universal_api.md#clientparam) for details.|

- <a id="li183183578215"></a>**Image Structuring Description Prompt (`IMG_TO_TEXT_PROMPT`)**

```text
IMG_TO_TEXT_PROMPT = '''Given an image containing a table or figure, please provide a structured and detailed
description in chinese with two levels of granularity:

  Coarse-grained Description:
  - Summarize the overall content and purpose of the image.
  - Briefly state what type of data or information is presented, for example comparison, trend, or distribution.
  - Mention the main topic or message conveyed by the table or figure.

  Fine-grained Description:
  - Describe the specific details present in the image.
  - For tables: List the column and row headers, units, and any notable values, patterns, or anomalies.
  - For figures, such as plots or charts: Explain the axes, data series, legends, and any significant trends, outliers,
  or data points.
  - Note any labels, captions, or annotations included in the image.
  - Highlight specific examples or noteworthy details.

  Deliver the description in a clear, organized, and reader-friendly manner, using bullet points or paragraphs
  as appropriate, answer in chinese'''
```

**Example**

```python
from mx_rag.llm import Img2TextLLM, LLMParameterConfig
from mx_rag.utils import ClientParam
from PIL import Image
import io
import base64

vlm = Img2TextLLM(base_url="https://{ip}:{port}/openai/v1/chat/completions",
                   model_name="Qwen2.5-VL-7B-Instruct",
                   llm_config=LLMParameterConfig(max_tokens=512),
                   client_param=ClientParam(ca_file="/path/to/ca.crt")
                   )
# Generate the base64-encoded image
with Image.open("/path/to/image.jpeg") as img:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

image_url = {"url": f"data:image/jpeg;base64,{img_base64}"}
res = vlm.chat(image_url=image_url)
print(res)
```

#### `chat`

**Description**

Interact with the VLM service to obtain the inference result of the VLM model.

**Prototype**

```python
def chat(image_url, prompt, sys_messages, role, llm_config)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|image_url|dict|Required|Dictionary that contains base64-encoded image data. The key is `url`, and the value is a string with `img_base64` as a variable. Example: `{"url": f"data:image/jpeg;base64,{image_base64}"}`. `image_base64` is the base64-encoded image data. Length range: [1, 4 \* 1024 \* 1024].|
|sys_messages|List[dict]|Optional|System messages. The list can contain up to 16 items. Each dictionary can contain up to 16 items. Each dictionary key string can be up to 16 characters. Each value string can be up to 4 \* 1024 \* 1024 characters. The default value is `None`.|
|role|str|Optional|Role of the inference request message. Length range: `[1, 16]`. The default value is `user`.|
|llm_config|LLMParameterConfig|Optional|Parameters for calling the LLM. See [LLMParameterConfig](#llmparameterconfig) for details.|

**Returns**

|Data Type|Description|
|--|--|
|str|Summary description of the image content from the VLM.|
