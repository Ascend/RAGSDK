# encoding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

PROMPTS = {}


PROMPTS[
    "DEFAULT_TOTAL_INFO_TMPL"
] = """
        【任务】：下面提供了一段文本。
        首先，请根据提供的文本，提取出所有包含头实体-关系-尾实体的三元组，请处理全部三元组，不要省略部分结果。
        其次，针对每个三元组中的头实体和尾实体，根据给定的文本上下文信息，给出一个描述的概念词。
        最后，根据文本，生成不超过20个字的总结.
        返回结果参考生成格式，直接生成json格式，不需要添加额外信息或解释。

        【文本】：{text}

        【生成格式】：
        {{"Triplets": [["头实体", "关系", "尾实体"]], "Entity": {{"实体": "实体概念"}}, "Summary": XX}}

        【示例】：
        给定文本："科学家们发现了一颗新的行星"
        返回结果为：
        {{"Triplets": [["科学家", "发现", "新的行星"]], "Entity": {{"科学家": "人", "新的行星": "科学成果"}}, "Summary": "科学家发现行星。"}}
"""

PROMPTS[
    "DEFAULT_TOTAL_INFO_TMPL_EN"
] = """
    # Goal
    Given a text document, identify all entities from the text and all relationships among the identified entities, and generate a comprehensive summary of the text.

    # Steps
    1. Identify all entities. For each identified entity, extract the following information:
        - entity_name: Name of the entity, capitalized
        - entity_concept: Give a concept word based on the given text context for each entity.
        
    2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
        For each pair of related entities, extract the following information:
        - source_entity: name of the source entity, as identified in step 1
        - target_entity: name of the target entity, as identified in step 1
        - relationship: the relationship between the source entity and the target entity
    3. Generate a comprehensive summary of the text of no more than 30 words.
    4. Format each output as a JSON entry with the following format:
        {{"Triplets": [["source_entity", "relationship", "target_entity"]], "Entity": {{"entity_name": "entity_concept"}}, "Summary": "summary sentence"}}

    The output is directly generated in JSON format. No additional information or explanation is required.
    
    # Examples
    ######################
    text: "Scientists have found a new shinning star recently."
    output:
    {{"Triplets": [["Scientists", "found", "a new star"]], "Entity": {{"Scientists": "Person", "a new star": "Scientific discovery"}}, "Summary": "Scientists found a star."}}

    #Real Data
    ######################
    text: {text}
    ######################
    output:
"""

PROMPTS[
    "DEFAULT_TOTAL_INFO_TMPL_WITH_ENTITY_TYPES"
] = """
        【任务】：下面提供了一段文本。
        请根据给定的实体类型，参考给出的示例，根据提供的文本提取出给定类型的可能实体、实体属性及实体间的关系，请处理全部三元组，不要省略部分结果。
        然后，根据文本，生成一句不超过20个字的总结。
        返回结果参考生成格式，直接生成json格式，不需要添加额外信息或解释。

        【注意】：
        1.只提取给定实体类型的实体，不在给定类型里的实体和相关三元组不要提取；

        【生成格式】：
        {{"Triplets": [["头实体", "关系", "尾实体"]], "Entity": {{"实体": "实体概念"}}, "Summary": "总结句"}}

        【示例】：
        [文本]：
        小明是李刚的儿子，今年17岁，居住在长亭苑。
        [实体类型]：人物，年龄，地点
        [返回结果]：
        {{
        "Triplets":[["李刚", "父亲", "小明"],["小明", "年龄", "17岁"],["小明", "居住", "长亭苑"],], 
        "Entity":{{"李刚": "人物", "小明": "人物", "17岁": "年龄", "长亭苑": "地点"}},
        "Summary": "小明是李刚17岁的儿子。"
        }}

        【文本】：{text}

        【实体类型】：{entity_types}

        【返回结果】：
"""

PROMPTS[
    "DEFAULT_TOTAL_INFO_TMPL_WITH_ENTITY_TYPES_EN"
] = """
        Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

        -Steps-
        1. Identify all entities. For each identified entity, extract the following information:
        - entity_name: Name of the entity, capitalized
        - entity_type: One of the following types: [{entity_types}]
        
        2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
        For each pair of related entities, extract the following information:
        - source_entity: name of the source entity, as identified in step 1
        - target_entity: name of the target entity, as identified in step 1
        - relationship: the relationship between the source entity and the target entity
        
        3. Generate a comprehensive summary of the text of no more than 30 words.
        
        4. Format each output as a JSON entry with the following format:
            {{"Triplets": [["source_entity", "relationship", "target_entity"]], "Entity": {{"entity_name": "entity_type"}}, "Summary": "summary sentence"}}
        
        The output is directly generated in JSON format. No additional information or explanation is required.
        
        ######################
        -Examples-
        ######################
        text: "Xiao Ming is son of Li Gang. He is 17 years old and lives in Beijing."
        entity_types: ["person", "age", "location"]
        output:
        {{"Triplets": [["Li Gang", "Father", "Xiao Ming"], ["Xiaoming", "Age", "17 years old"],["Xiao Ming", "Lives", "Beijing"]], 
        "Entity": {{"Li Gang": "person", "Xiao Ming": "person", "17 years old": "age", "Beijing": "location"}}, 
        "Summary": "Xiao Ming is Li Gang's son, 17 years old."}}
    
        #Real Data
        ######################
        entity_types: {entity_types}
        text: {text}
        ######################
        output:
"""


PROMPTS[
    "GENERATOR"
] = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an  assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer as concise and accurate as possible.
        Do NOT repeat the question or output any other words<|eot_id|><|start_header_id|>user<|end_header_id|>
        Context: {context} 
        Question: {question} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

PROMPTS[
    "MULTIPLE_ROUNDS"
] = """
        请根据历史对话以及用户当前的问题，对用户当前的问题进行改写，并根据改写后的问题分解出子问题，以便于信息检索。
    
        生成格式如下：
        {{
        "new_question":改写后的问题,
        "sub_questions":根据改写后的问题分解出的子问题,
        }}
            
        历史对话如下：
        {history}
        
        用户提问如下：
        {question}
    
        要求：
            1.根据历史对话将用户问题中的指代词进行替换，改写后的问题尽可能简洁。
            2.根据改写后的问题分解出的子问题应尽可能少,不要进行联想。
    
        请开始
"""

PROMPTS[
    "NOT_USING_RETRIEVAL"
] = """
    请根据已知信息，简洁和专业的来回答问题, 不允许在答案中添加编造成分，答案请使用中文。
    
    已知信息:
    {context}
    
    问题:
    {query}
    
    注意：
    1.已知信息中可能会存在一些不相关的知识，请从相关性和可用性等多个维度对已知信息的内容进行分析，只保留最相关、最优质的内容；
    
    请给出问题的答案
"""

PROMPTS[
    "RETRIEVAL_REWRITING"
] = """
    你是一个信息检索机器人，请提取用户问题中的关键词用于检索相关信息。

    生成json格式如下：
    {{
    "keywords":["xxx", ... , "xxx"]
    }}
        
    用户提问如下：
    {question}

    回答：
"""
