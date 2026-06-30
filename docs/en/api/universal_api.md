# General

## `ClientParam`

### Class Functionality

**Description**

Configuration parameters for connecting to the server.

**Function Prototype**

```python
from mx_rag.utils import ClientParam
ClientParam(use_http, ca_file, crl_file, timeout, response_limit_size)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|use_http|bool|Optional|Specifies whether the client can use the HTTP protocol. The default value is `False`, which means the client uses HTTPS. <br>HTTP poses security risks. Therefore, you are advised to use HTTPS.|
|ca_file|str|Optional|Server root certificate. The default value is `""`. The path length cannot exceed 1024 characters. The path cannot be a symbolic link, `..` is not allowed, and the file size cannot exceed 1 MB.|
|crl_file|str|Optional|Server certificate revocation list. The default value is `""`. The path length cannot exceed 1024 characters. The path cannot be a symbolic link, `..` is not allowed, and the file size cannot exceed 1 MB.|
|timeout|int|Optional|Response timeout for the server. The value range is `(0, 600]`. The default value is 60. Unit: seconds.|
|response_limit_size|int|Optional|Maximum number of bytes in the server response that the client accepts. The value range is (0, 10 MB]. The default value is 1 MB.|

> [!NOTE]
>When creating the client, check `use_http`. If HTTPS is enabled, `ca_file` is required. If only `ca_file` is provided, the client creates a TLS/SSL context with one-way authentication. If HTTPS is not enabled, the client uses the default TLS/SSL context.

**Example**

```python
from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils import ClientParam
llm = Text2TextLLM(base_url="https://{ip}:{port}/v1/chat/completions",
                   model_name="qianwen-7b",
                   llm_config=LLMParameterConfig(max_tokens=512),
                   client_param=ClientParam(ca_file="/path/to/ca.crt")
                   )
res = llm.chat("请介绍下北京")
print(res)
for res in llm.chat_streamly("请介绍下北京"):
    print(res)
```

## `Lang`

### Class Functionality

Language enumeration class.

Currently, two languages are supported, English (`EN`) and Chinese (`CH`).

- `EN`: Set the working language to English.
- `CH`: Set the working language to Chinese.

**function Prototype**

```python
from mx_rag.utils import Lang
class Lang(Enum):
    EN: str = 'en'
    CH: str = 'ch'
```
