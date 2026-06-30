# Knowledge Graph

Implements a knowledge graph-based retrieval-augmented generation (RAG) process. It combines large language models (LLMs), graph databases such as NetworkX or openGauss, vector retrieval, reranking, and other techniques to support structured knowledge extraction from documents, graph construction, concept clustering, vector-based retrieval, and multi-hop reasoning. This module is suitable for complex knowledge question answering, enterprise knowledge management, intelligent retrieval, and similar scenarios.

The knowledge graph uses an LLM and structured prompts to perform relation extraction and conceptualize nodes and relations. To make it easier for you to customize the default prompts, the following Chinese and English versions are provided for reference:

**Chinese default prompts:**

- **Triple extraction prompt (`TRIPLE_INSTRUCTIONS_CN`):**

    ```text

    TRIPLE_INSTRUCTIONS_CN = {
        "entity_relation": """
    ## 目标
    请从以下文本中提取所有重要实体及其关系，并严格遵守以下规则：
    ## 要求
    1. 实体必须为名词，尽量简洁；
    2. 关系必须为一个动词，准确描述“头实体”与“尾实体”之间的具体联系，且不得重复头、尾实体的字面信息；
    3. 头实体与尾实体均不得为“是”，不得使用代词；
    4. 实体和关系不能为空字符串，不能为仅包含标点符号的字符串；
    5. 输出必须采用下列 JSON 格式，禁止添加、删除或修改任何字段：
    [
        {
            "头实体": "{名词}",
            "关系": "{动词}",
            "尾实体": "{名词}"
        }
    ]
    ## 示例
    [
        {
            "头实体": "中国",
            "关系": "首都",
            "尾实体": "北京",
        },
        {
            "头实体": "小狗",
            "关系": "喜欢",
            "尾实体": "骨头",
        },
        {
            "头实体": "毛泽东",
            "关系": "父亲",
            "尾实体": "毛岸英",
        },
        {
            "头实体": "中国船舶工业物资云贵有限公司",
            "关系": "成立",
            "尾实体": "1990月05月31日",
        },
        {
            "头实体": "公司",
            "关系": "地址",
            "尾实体": "云南省昆明市",
        },
        {
            "头实体": "公司",
            "关系": "经营",
            "尾实体": "电子器件",
        },
        {
            "头实体": "1999年",
            "关系": "早于",
            "尾实体": "2000年",
        },
        {
            "头实体": "2001年",
            "关系": "晚于",
            "尾实体": "2000年",
        }
    ]
    ## 待分析文本
    """,
        "event_entity": """
    ## 目标
    请对以下段落逐句进行事件抽取，并识别每个事件所涉及的全部实体。
    ## 要求
    1. 一句视为一个独立事件，保留原句，不做任何省略；
    2. 列出每个事件直接参与的所有实体，不重复、不遗漏；
    3. 输出严格使用下方 JSON 格式，不允许添加或删减字段。
    ## JSON 格式
    [
        {
            "事件": "{原句}",
            "实体": ["实体1", "实体2", "..."]
        }
    ]
    ## 待分析文本
    """,
        "event_relation": """
    ## 目标
    请对以下段落逐句抽取事件，并识别它们之间的时间或因果关系。
    ## 要求
    1. 一句视为一个独立事件，保留原句，不做任何省略。
    2. 仅使用指定关系类型：在之前、在之后、同时、因为、结果。
    3. 每个三元组中的“头事件”与“尾事件”均须为段落中完整原句，且语义对应具体、可独立理解。
    4. “头事件”和“尾事件”不能为空字符串，且不能重叠；
    5. 关系不能为空字符串；
    6. 输出严格使用下方 JSON 格式，不允许添加、删减或省略任何字段。
    ## JSON 格式
    [
        {
            "头事件": "{事件1完整原句}",
            "关系": "{在之前|在之后|同时|因为|结果}",
            "尾事件": "{事件2完整原句}"
        }
    ]
    ## 待分析文本
    """,
    }
    ```

- **Entity conceptualization prompt (`ENTITY_PROMPT_CN`):**

    ```text
    ENTITY_PROMPT_CN = '''
    ## 目标
    给定一个实体及其背景，提供多个短语来表示该实体的抽象概念。
    ## 输出格式
    短语1, 短语2, 短语3,...
    ## 要求
    * 短语应准确代表实体，可以是其类型或相关概念。
    * 短语包含1-2个词。
    * 短语不能包含空格和换行符，不能包含标点符号，但可以包含连字符。
    * 严格遵循输出格式，不添加任何额外字符。
    * 尽可能提供3到10个不同抽象层次的短语。
    * 不重复使用与实体或已有短语相同的词语。
    * 如果无法生成更多短语，请立即停止。
    ## 示例
    实体：灵魂
    背景：在BFI伦敦电影节首映，成为皮克斯票房最高的影片
    概念：电影, 影片
    实体：ThinkPad X60
    背景：理查德·斯托曼宣布他在ThinkPad X60上使用Trisquel
    概念：ThinkPad, 笔记本, 机器, 设备, 硬件, 电脑, 品牌
    实体：哈里·卡拉汉
    背景：欺骗另一个抢劫犯，折磨天蝎座
    概念：人, 美国人, 角色, 警察, 侦探
    实体：黑山学院
    背景：由约翰·安德鲁·赖斯创立，吸引了教师
    概念：学院, 大学, 学校, 文理学院
    ## 待分析文本
    实体：[ENTITY]
    背景：[CONTEXT]
    概念：
    '''
    ```

- **Event conceptualization prompt (`EVENT_PROMPT_CN`):**

    ```text
    EVENT_PROMPT_CN = '''
    ## 目标
    给定一个事件，提供多个短语来表示该事件的抽象概念。
    ## 输出格式
    短语1, 短语2, 短语3,...
    ## 要求
    * 短语应准确代表事件，可以是其类型或相关概念。
    * 短语包含1-2个词。
    * 短语不能包含空格和换行符，不能包含标点符号，但可以包含连字符。
    * 严格遵循输出格式，不添加任何额外字符。
    * 尽可能提供3到10个不同抽象层次的短语。
    * 不重复使用与事件或已有短语相同的词语。
    * 如果无法生成更多短语，请立即停止。
    ## 示例
    事件：一名男子隐居于山林
    概念：隐居, 放松, 逃避, 自然, 孤独
    事件：一只猫追逐猎物进入它的藏身之处
    概念：狩猎, 逃避, 捕食, 躲藏, 潜行
    事件：山姆和他的狗玩耍
    概念：放松的活动, 爱抚, 玩耍, 联结, 友谊
    事件：中国船舶工业物资云贵有限公司，成立日期：1990年3月31日成立，住所：云南省昆明市
    概念：公司成立, 公司地点, 公司住所, 省, 市, 成立时间, 年月日
    ## 待分析文本
    事件：[EVENT]
    概念：
    '''
    ```

- **Relation conceptualization prompt (`RELATION_PROMPT_CN`):**

    ```text
    RELATION_PROMPT_CN = '''
    ## 目标
    给定一个关系，提供多个短语来表示该关系的抽象概念。
    ## 输出格式
    短语1, 短语2, 短语3,...
    ## 要求
    * 短语应准确代表关系，可以是其类型或相关概念。
    * 短语包含1-2个词。
    * 短语不能包含空格和换行符，不能包含标点符号，但可以包含连字符。
    * 严格遵循输出格式，不添加任何额外字符。
    * 尽可能提供3到10个不同抽象层次的短语。
    * 不重复使用与关系或已有短语相同的词语。
    * 如果无法生成更多短语，请立即停止。
    ## 示例
    关系：参与
    概念：成为一部分, 参加, 投入, 涉及
    关系：被包括在内
    概念：加入, 成为一部分, 成为成员, 成为组成部分
    ## 待分析文本
    关系：[RELATION]
    概念：
    '''
    ```

**English Default Prompts:**

    ```text
    TRIPLE_INSTRUCTIONS_EN = {
        "entity_relation": """Given a passage, summarize all the important entities and the relations between them in
        a concise manner. Relations should briefly capture the connections between entities, without repeating information
        from the head and tail entities. The entities should be as specific as possible. Exclude pronouns from
        being considered as entities. The output should strictly adhere to the following JSON format:
        [
            {
                "Head": "{a noun}",
                "Relation": "{a verb}",
                "Tail": "{a noun}"
            },
            {
                "Head": "China",
                "Relation": "Capital",
                "Tail": "Beijing"
            },
            {
                "Head": "Dog",
                "Relation": "like",
                "Tail": "bone"
            },
            {
                "Head": "Mao Zedong",
                "Relation": "Father",
                "Tail": "Mao Anying"
            },
            {
                "Head": "China Shipbuilding Materials Yungui Co., Ltd.",
                "Relation": "Established",
                "Tail": "May 31, 1990"
            },
            {
                "Head": "Company",
                "Relation": "Address",
                "Tail": "Kunming City, Yunnan Province"
            },
            {
                "Head": "Company",
                "Relation": "Operation",
                "Tail": "Electronics"
            },
            {
                "Head": "Year 1999",
                "Relation": "Before",
                "Tail": "Year 2000"
            },
            {
                "Head": "Year 2001",
                "Relation": "After",
                "Tail": "Year 2000"
            },
        ]""",
        "event_entity": """Please analyze and summarize the participation relations between the events and entities
        in the given paragraph. Each event is a single independent sentence. Additionally, identify all the entities
        that participated in the events. Do not use ellipses. Please strictly output in the following JSON format:
        [
            {
                "Event": "{a simple sentence describing an event}",
                "Entity": ["entity 1", "entity 2", "..."]
            }...
        ] """,
        "event_relation": """Please analyze and summarize the relationships between the events in the paragraph.
        Each event is a single independent sentence. Identify temporal and causal relationships between the events using the following types: before, after, at the same time, because, and as a result. Each extracted triple should be specific, meaningful, and able to stand alone.  Do not use ellipses.  The output should strictly adhere to the following JSON format:
        [
            {
                "Head": "{a simple sentence describing the event 1}",
                "Relation": "{temporal or causality relation between the events}",
                "Tail": "{a simple sentence describing the event 2}"
            }...
        ]"""
    }
    ```

- **Entity conceptualization prompt (`ENTITY_PROMPT_CN`):**

    ```text
    ENTITY_PROMPT_EN = '''I will give you an ENTITY. You need to give several phrases containing 1-2 words for the
                ABSTRACT ENTITY of this ENTITY.
                You must return your answer in the following format: phrases1, phrases2, phrases3,...
                You can't return anything other than answers.
                These abstract intention words should fulfill the following requirements.
                1. The ABSTRACT ENTITY phrases can well represent the ENTITY, and it could be the type of the ENTITY or
                the related concepts of the ENTITY.
                2. Strictly follow the provided format, do not add extra characters or words.
                3. Write at least 3 or more phrases at different abstract level if possible.
                4. Do not repeat the same word and the input in the answer.
                5. Stop immediately if you can't think of any more phrases, and no explanation is needed.
                ENTITY: Soul
                CONTEXT: premiered BFI London Film Festival, became highest-grossing Pixar release
                Your answer: movie, film
                ENTITY: ThinkPad X60
                CONTEXT: Richard Stallman announced he is using Trisquel on a ThinkPad X60
                Your answer: ThinkPad, laptop, machine, device, hardware, computer, brand
                ENTITY: Harry Callahan
                CONTEXT: bluffs another robber, tortures Scorpio
                Your answer: person, American, character, police officer, detective
                ENTITY: Black Mountain College
                CONTEXT: was started by John Andrew Rice, attracted faculty
                Your answer: college, university, school, liberal arts college
                EVENT: 1st April
                CONTEXT: Utkal Dibas celebrates
                Your answer: date, day, time, festival
                ENTITY: [ENTITY]
                CONTEXT: [CONTEXT]
                Your answer:
                '''
    ```

- **Event conceptualization prompt (`EVENT_PROMPT_CN`):**

    ```text
    EVENT_PROMPT_EN = '''I will give you an EVENT. You need to give several phrases containing 1-2 words for the
                ABSTRACT EVENT of this EVENT.
                You must return your answer in the following format: phrases1, phrases2, phrases3,...
                You can't return anything other than answers.
                These abstract event words should fulfill the following requirements.
                1. The ABSTRACT EVENT phrases can well represent the EVENT, and it could be the type of the EVENT or the related concepts of the EVENT.
                2. Strictly follow the provided format, do not add extra characters or words.
                3. Write at least 3 or more phrases at different abstract level if possible.
                4. Do not repeat the same word and the input in the answer.
                5. Stop immediately if you can't think of any more phrases, and no explanation is needed.
                EVENT: A man retreats to mountains and forests
                Your answer: retreat, relaxation, escape, nature, solitude

                EVENT: A cat chased a prey into its shelter
                Your answer: hunting, escape, predation, hiding, stalking
                EVENT: Sam playing with his dog
                Your answer: relaxing event, petting, playing, bonding, friendship
                EVENT: [EVENT]
                Your answer:
                '''
    ```

- **Relation conceptualization prompt (`RELATION_PROMPT_CN`):**

    ```text
    RELATION_PROMPT_EN = '''I will give you an RELATION. You need to give several phrases containing 1-2 words for
                the ABSTRACT RELATION of this RELATION.
                You must return your answer in the following format: phrases1, phrases2, phrases3,...
                You can't return anything other than answers.
                These abstract intention words should fulfill the following requirements.
                1. The ABSTRACT RELATION phrases can well represent the RELATION, and it could be the type of the RELATION
                or the simplest concepts of the RELATION.
                2. Strictly follow the provided format, do not add extra characters or words.
                3. Write at least 3 or more phrases at different abstract level if possible.
                4. Do not repeat the same word and the input in the answer.
                5. Stop immediately if you can't think of any more phrases, and no explanation is needed.

                RELATION: participated in
                Your answer: become part of, attend, take part in, engage in, involve in
                RELATION: be included in
                Your answer: join, be a part of, be a member of, be a component of
                RELATION: [RELATION]
                Your answer:
                '''
    ```

## `GraphRAGPipeline`

### Class Description

**Description**

Provides a unified entry point for knowledge graph creation and retrieval.

**Prototype**

```python
from mx_rag.graphrag import GraphRAGPipeline

GraphRAGPipeline(work_dir, llm, embedding_model, dim, rerank_model, graph_type,graph_name, encrypt_fn,decrypt_fn,kwargs)
```

**Input Parameters**

| Parameter | Data Type | Optional/Required | Description |
|--|--|--|--|
| work_dir | str | Required | The knowledge graph working directory. It must have at least 5 GB of free space. The generated graph JSON intermediate files are stored here, and the corresponding vector data is also stored here if MindFAISS is used. <br>The path cannot be relative, its length cannot exceed 1024, it cannot be a symbolic link, and it must not contain `..`. <br>The path cannot be in the following list: [`/etc`, `/usr/bin`, `/usr/lib`, `/usr/lib64`, `/sys/`, `/dev/`, `/sbin`, `/tmp`]. |
| llm | Text2TextLLM | Required | The LLM interface instance. |
| embedding_model | Embeddings | Required | A subclass of `langchain_core.embeddings.Embeddings`. Supported values include: <li>`mx_rag.embedding.local.TextEmbedding`</li><li>`mx_rag.embedding.service.TEIEmbedding`</li> |
| dim | int | Required | The vector dimension generated by the embedding model. The value range is `[1, 1024 * 1024]`. |
| rerank_model | Reranker | Optional | A subclass of `mx_rag_reranker.Reranker`. The default value is `None`. Supported values include: <li>`mx_rag.reranker.local.LocalReranker`</li><li>`mx_rag.reranker.service.TEIReranker`</li> |
| graph_type | str | Optional | The graph database type. The default value is `"networkx"`. Supported values are [`networkx`, `opengauss`]. |
| graph_name | str | Optional | The knowledge graph name. The default value is `"graph"`. The value range is `[1, 255]`, and it can contain only identifiers. |
| encrypt_fn | Callable | Optional | The callback that encrypts the contents of the JSON file generated by calling [build_graph](#build_graph). Ensure that you provide the correct encryption method and keep it secure. The return value is the encrypted string. <br>If the uploaded documents contain personal data such as bank card numbers, ID card numbers, passport numbers, or passwords, configure this parameter to protect personal data. |
| decrypt_fn | Callable | Optional | The callback that decrypts and reads `"{graph_name}.json"` during retrieval when `graph_type` is `"networkx"`. Ensure that you provide the correct decryption method and keep it secure. The return value is the decrypted string. |
| kwargs | Dict | Optional | Extended parameters: <li>`age_graph`: When the graph database type is openGauss, this parameter is required. The type is `openGaussAGEGraph`, which is an openGauss graph database connection instance.</li><li>`devs`: Specifies the NPU device. It is a list with exactly one element of type `list[int]`.</li><li>`node_vector_store`: Vector database used to store vectorized nodes for similar node search. The default value is `None`, in which case MindFAISS is used as the vector database.</li><li>`conceptualize`: Whether to perform concept clustering. The default value is `False`. When clustering is disabled, `concept_vector_store` does not take effect.</li><li>`concept_vector_store`: Vector database used to store vectorized concepts for similar concept search when clustering concepts. The default value is `None`, in which case MindFAISS is used as the vector database.</li><br>`age_graph` is passed in under user control. Please use a secure connection method. |

**Returns**

`GraphRAGPipeline` object.

**Usage Example<a name="section8509453104117"></a>**

```python
import getpass
from paddle.base import libpaddle  # fix std::bad_alloc
from langchain_opengauss import OpenGaussSettings, openGaussAGEGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from mx_rag.chain.single_text_to_text import GraphRagText2TextChain
from mx_rag.document import LoaderMng
from mx_rag.embedding.local import TextEmbedding
from mx_rag.graphrag import GraphRAGPipeline
from mx_rag.llm import LLMParameterConfig, Text2TextLLM
from mx_rag.reranker.local import LocalReranker
from mx_rag.utils import ClientParam
work_dir = "test_pipeline"
llm = Text2TextLLM(
    base_url="https://x.x.x.x:port/v1/chat/completions",
    model_name="model_name",
    llm_config=LLMParameterConfig(max_tokens=64 * 1024, temperature=0.6, top_p=0.9),
    client_param=ClientParam(timeout=180, ca_file="/path/to/ca.crt"),
)
rerank_model = LocalReranker("/data/models/bge-reranker-v2-m3/", 0, 20, False)
embedding_model = TextEmbedding.create(model_path="/data/models/bge-large-en-v1.5")
data_load_mng = LoaderMng()
data_load_mng.register_loader(TextLoader, [".txt"])
data_load_mng.register_splitter(
    RecursiveCharacterTextSplitter,
    [".txt"],
    dict(chunk_size=512, chunk_overlap=20)
)
graph_name = "hotpotqa"
graph_type = "opengauss"

conf = OpenGaussSettings(user="gaussdb",
                         password=getpass.getpass(),
                         host="x.x.x.x",
                         port="x",
                         database="postgres")
age_graph = openGaussAGEGraph(graph_name, conf,
                              sslmode="verify-ca",
                              sslcert="client.crt",
                              sslkey="client.key",
                              sslrootcert="cacert.pem")
pipeline = GraphRAGPipeline(work_dir, llm, embedding_model, 1024, rerank_model, graph_name=graph_name,
                            age_graph=age_graph)
pipeline.upload_files(["./test_graph/hotpotqa.500.txt"], data_load_mng)
pipeline.build_graph()
question = "Which case was brought to court first Miller v. California or Gates v. Collier ?"
contexts = pipeline.retrieve_graph(question)
text2text_chain = GraphRagText2TextChain(
    llm=llm,
    retriever=pipeline.as_retriever(),
    reranker=rerank_model)
result = text2text_chain.query(question)
print(f"#contexts: {len(contexts)}")
print(contexts)
print(result)
```

### `upload_files`

**Description**

Uploads the list of documents required to build the knowledge graph.

**Prototype**

```python
def upload_files(file_list, loader_mng)
```

**Input Parameters**

| Parameter | Data Type | Optional/Required | Description |
|--|--|--|--|
| file_list | list | Required | The document list. A batch of documents supports only one language type. Uploading too many documents at once will slow down knowledge graph construction. The number of documents is limited to `[1, 100]`. <br>The path length of a single document must be in the range `[1, 1024]`. The document path cannot be a symbolic link and must not contain `..`. Each document must be no larger than 10 GB. |
| loader_mng | LoaderMng | Required | The manager object that provides document parsing functions. See LoaderMng（需补充链接）for the data type. |

**Returns**

None

### `build_graph`

**Description**

Creates text node indexes and generates the knowledge graph for the corresponding text.

**Prototype**

```python
def build_graph(lang, **kwargs)
```

**Input Parameters**

| Parameter | Data Type | Optional/Required | Description |
|--------|------|------|------|
| lang | Lang | Optional | The language of the corpus. The default value is `Lang.EN`, which indicates English corpus. |
| kwargs | dict | Optional | Extended parameters: <li>`max_workers`: The number of threads used to build the knowledge graph. The default value is `5`.</li><li>`batch_size`: The batch size for node vectorization, retrieval, and other operations. The default value is `32`.</li><li>`top_k`: When clustering graph node concepts, the number of most similar concepts returned by vector retrieval. The default value is `5`, and the value range is `[1, 100]`.</li><li>`threshold`: The vector similarity threshold. Results below this value are filtered out. The default value is `0.3`, and the value range is `[0.0, 1.0]`.</li><li>`triple_instructions`: The instructions used to guide the LLM to extract relations from documents. The type is `dict`. The default value is `None`. In that case, the default value is selected according to the language, with `TRIPLE_INSTRUCTIONS_CN` for Chinese and `TRIPLE_INSTRUCTIONS_EN` for English. You can override the default extraction instructions by providing a dictionary. The dictionary must include the following keys:<ul><li>`entity_relation`: The corresponding value defines the instructions for entity-relation extraction. The type is `str`, and the length range is `[1, 1048576]`.</li><li>`event_entity`: The corresponding value defines the instructions for event-entity extraction. The type is `str`, and the length range is `[1, 1048576]`.</li><li>`event_relation`: The corresponding value defines the instructions for event-relation extraction. The type is `str`, and the length range is `[1, 1048576]`.<br>Each key maps to the instructions for a specific extraction task.</li></ul></li><li>`conceptualizer_prompts`: Prompts used to guide the LLM in conceptualization. The type is `dict`. The default value is `None`. You can override the default conceptualization prompts by providing a dictionary. The dictionary must include the following keys:<ul><li>`entity`: The corresponding value defines the prompt for conceptualizing graph entities. The type is `... [truncated]entity: 对应的值定义对图中实体进行概念化的提示， 字符串类型，长度范围为[1, 1048576]。当conceptualizer_prompts为None时将根据语言使用默认值（中文为ENTITY_PROMPT_CN，英文为ENTITY_PROMPT_EN）。</li><li>event: 定义对图中事件进行概念化的提示, 字符串类型，长度范围为[1, 1048576]。当conceptualizer_prompts为None时将根据语言使用默认值（中文为EVENT_PROMPT_CN，英文为EVENT_PROMPT_EN）。</li><li>relation: 定义对图中关系进行概念化的提示, 字符串类型，长度范围为[1, 1048576]。当conceptualizer_prompts为None时将根据语言使用默认值（中文为RELATION_PROMPT_CN，英文为RELATION_PROMPT_EN）。</li></ul></li>

**Returns**

None

After the method runs, the following intermediate files are generated in `work_dir`:

**Table 1**

| File | Description |
|--|--|
| `"{graph_name}.json"` | Stores the graph. When `graph_type` is `"networkx"`, retrieval loads the graph from this file. |
| `"{graph_name}_relations.json"` | Stores entity relation information. |
| `"{graph_name}_concepts.json"` | Stores concept information. |
| `"{graph_name}_synset.json"` | Stores the category information after concept clustering. |
| `"{graph_name}_node_vectors.index"` | The vector index file for entities. |
| `"{graph_name}_concept_vectors.index"` | The vector index file for concepts. |

### `retrieve_graph`

**Description**

Retrieves and returns relevant document chunks.

**Prototype**

```python
def retrieve_graph(question, **kwargs)
```

**Input Parameters**

| Parameter | Data Type | Optional/Required | Description |
|--|--|--|--|
| question | str | Required | The user question. The string length range is `[1, 1000 * 1000]`. |
| kwargs | dict | Optional | Extended parameters: <li>`use_text`: A boolean value. The default value is `True`, which means that when retrieving the subgraph, only the text contained in text-type nodes is used to build the context.</li><li>`batch_size`: An integer. The default value is `4`. It specifies the batch size used when vectorizing nodes. The range is `[1, 1024]`.</li><li>`similarity_tail_threshold`: The vector similarity threshold. The default value is `0.0`. Results below this value are filtered out. The range is `[0.0, 1.0]`.</li><li>`retrieval_top_k`: An integer. The default value is `40`. It specifies the top `k` value used when retrieving similar nodes from the node vector database based on entities. The range is `[1, 1000]`.</li><li>`reranker_top_k`: The top `k` value required by the reranker. The default value is `20`. The range is `[1, 1000]`.</li><li>`subgraph_depth`: An integer. The default value is `2`. It specifies the maximum exploration depth for graph retrieval. The range is `[1, 5]`.</li> |

**Returns**

| Data Type | Description |
|--|--|
| `List[str]` | Retrieved context chunks. |

### `as_retriever`

**Description**

Returns a retriever.

**Prototype**

```python
def as_retriever(**kwargs)
```

**Input Parameters**

| Parameter | Data Type | Optional/Required | Description |
|--|--|--|--|
| kwargs | dict | Optional | Extended parameters: <li>`use_text`: A boolean value. The default value is `True`, which means that when retrieving the subgraph, only the text contained in text-type nodes is used to build the context.</li><li>`batch_size`: An integer. The default value is `4`. It specifies the batch size used when vectorizing nodes. The range is `[1, 1024]`.</li><li>`similarity_tail_threshold`: The vector similarity threshold. The default value is `0.0`. Results below this value are filtered out. The range is `[0.0, 1.0]`.</li><li>`retrieval_top_k`: An integer. The default value is `40`. It specifies the top `k` value used when retrieving similar nodes from the node vector database based on entities. The range is `[1, 1000]`.</li><li>`reranker_top_k`: The top `k` value required by the reranker. The default value is `20`. The range is `[1, 1000]`.</li><li>`subgraph_depth`: An integer. The default value is `2`. It specifies the maximum exploration depth for graph retrieval. The range is `[1, 5]`.</li> |

**Returns**

| Data Type | Description |
|--|--|
| `GraphRetriever` | This retriever inherits from `langchain_core.retrievers.BaseRetriever`. |

## `GraphEvaluator`

### Class Description

**Description**

Evaluates the quality of a knowledge graph.

**Prototype**

```python
from mx_rag.graphrag import GraphEvaluator

GraphEvaluator(llm, llm_config)
```

**Input Parameters**

| Parameter | Data Type | Optional/Required | Description |
|--|--|--|--|
| llm | Text2TextLLM | Required | The LLM interface instance. |
| llm_config | LLMParameterConfig | Required | See [LLMParameterConfig](./llm_client.md#llmparameterconfig) for details. |

**Returns**

`GraphEvaluator` object.

**Usage Example<a name="section8509453104117"></a>**

```python
import json
from paddle.base import libpaddle
from mx_rag.graphrag.graph_evaluator import GraphEvaluator
from mx_rag.llm import Text2TextLLM, LLMParameterConfig
from mx_rag.utils import ClientParam
llm_config = LLMParameterConfig(temperature=0.5, top_p=0.8, max_tokens=8192)
llm = Text2TextLLM(
    base_url="https://ip:port/v1/chat/completions",
    model_name="Llama3-8B-Chinese-Chat",
    llm_config=llm_config,
    client_param=ClientParam(ca_file="/path/to/ca.crt", timeout=120),
)
graph_evaluator = GraphEvaluator(llm, llm_config)
relations_path = "/path/to/graph_relations.json"
with open(relations_path, "r", encoding="utf-8") as f:
    relations = json.load(f)
    graph_evaluator.evaluate(relations)
```

### `evaluate`

**Description**

Evaluates the triple relations extracted by the LLM.

**Prototype**

```python
def evaluate(relations)
```

**Input Parameters**

| Parameter | Data Type | Optional/Required | Description |
|--|--|--|--|
| relations | list[dict] | Required | The knowledge graph relation list. Each element is a dictionary and must include the `raw_text` key. The `entity_relations`, `event_entity_relations`, and `event_relations` keys are optional. The list length range is `[1, 50000]`, the maximum nesting depth is 5, and the text length limit is 4096. |

**Returns**

None. The function prints three groups of precision, recall, and F1 scores, corresponding to entity extraction, entity extraction in events, and event extraction. The higher the scores, the better the extraction quality.
