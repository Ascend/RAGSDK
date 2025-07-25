# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
TRIPLE_INSTRUCTIONS_EN = {
    "entity_relation": """Given a passage, summarize all the important entities and the relations between them in 
    a concise manner. Relations should briefly capture the connections between entities, without repeating information 
    from the head and tail entities. The entities should be as specific as possible. Exclude pronouns from 
    being considered as entities. The output should strictly adhere to the following JSON format:
    [
        {
            "Head": "{a noun}",
            "Relation": "{a verb}",
            "Tail": "{a noun}",
        },
        {
            "Head": "China",
            "Relation": "Capital",
            "Tail": "Beijing",
        },
        {
            "Head": "Dog",
            "Relation": "like",
            "Tail": "bone",
        },
        {
            "Head": "Mao Zedong",
            "Relation": "Father",
            "Tail": "Mao Anying",
        },
        {
            "Head": "China Shipbuilding Materials Yungui Co., Ltd.",
            "Relation": "Established",
            "Tail": "May 31, 1990",
        },
        {
            "Head": "Company",
            "Relation": "Address",
            "Tail": "Kunming City, Yunnan Province",
        },
        {
            "Head": "Company",
            "Relation": "Operation",
            "Tail": "Electronics",
        },
        {
            "Head": "Year 1999",
            "Relation": "Before",
            "Tail": "Year 2000",
        },
        {
            "Head": "Year 2001",
            "Relation": "After",
            "Tail": "Year 2000",
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

PASSAGE_START_EN = "Here is the passage. "

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
            Your answer: hunting, escape, predation, hidding, stalking

            EVENT: Sam playing with his dog
            Your answer: relaxing event, petting, playing, bonding, friendship

            EVENT: [EVENT]
            Your answer:
            '''

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

            ENTITY: Thinkpad X60
            CONTEXT: Richard Stallman announced he is using Trisquel on a Thinkpad X60
            Your answer: Thinkpad, laptop, machine, device, hardware, computer, brand

            ENTITY: Harry Callahan
            CONTEXT: bluffs another robber, tortures Scorpio
            Your answer: person, Amarican, character, police officer, detective

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

TRIPLE_INSTRUCTIONS_CN = {
    "entity_relation": """给定一段文字，简要总结所有重要的实体及其之间的关系。关系应该具体描述实体之间的联系，不要重复头实体和尾实体的信息。
    实体应尽可能简洁，头和尾实体不能为"是"。排除代词作为实体。输出必须严格遵循以下JSON格式：
    [
        {
            "头实体": "{一个名词}",
            "关系": "{一个动词}",
            "尾实体": "{一个名词}",
        }...
    ]
        例如
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
        },
    ]""",

    "event_entity": """请分析并总结给定段落中事件与实体之间的参与关系。每个事件都是一个独立的句子。此外，识别出所有参与事件的实体。不要使用省略号。输出必须严格遵循以下JSON格式：
    [
        {
            "事件": "{描述事件的简单句子}",
            "实体": ["实体1", "实体2", "..."]
        }...
    ] """,

    "event_relation": """请分析并总结段落中事件之间的关系。每个事件都是一个独立的句子。使用以下类型识别事件之间的时间和因果关系：
    在之前，在之后，同时，因为，结果。每个提取的三元组应具体、有意义，并且能够独立存在。不要使用省略号。输出必须严格遵循以下JSON格式：
    [
        {
            "头事件": "{描述事件1的简单句子}",
            "关系": "{事件之间的时间或因果关系}",
            "尾事件": "{描述事件2的简单句子}"
        }...
    ]"""

}


PASSAGE_START_CN = "以下是文本。"

EVENT_PROMPT_CN = '''我将给你一个事件。你需要给出几个包含1-2个词的短语来表示这个事件的抽象概念。
你必须按照以下格式返回你的答案：phrases1, phrases2, phrases3,...
除了答案之外，你不能返回任何其他内容。
这些抽象意图词应满足以下要求：

抽象事件短语能够很好地代表该事件，并且可以是事件的类型或与事件相关的概念。
严格遵循提供的格式，不要添加多余的字符或词语。
如果可能的话，至少写出三个或更多不同抽象层次的短语。
不要在答案中重复相同的词语和输入的词语。
如果再也想不出更多的短语，立即停止，不需要解释。
事件：一名男子隐居于山林
你的答案：隐居，放松，逃避，自然，孤独

事件：一只猫追逐猎物进入它的藏身之处
你的答案：狩猎，逃避，捕食，躲藏，潜行

事件：山姆和他的狗玩耍
你的答案：放松的活动，爱抚，玩耍，联结，友谊

事件：中国船舶工业物资云贵有限公司，成立日期：1990年3月31日成立，住所：云南省昆明市
你的答案：公司成立，公司地点，公司住所，省，市，成立时间，年月日

事件：[EVENT]
你的答案：
'''

ENTITY_PROMPT_CN = '''我将给你一个实体。你需要给出几个包含1-2个词的短语来表示这个实体的抽象概念。
你必须按照以下格式返回你的答案：phrases1, phrases2, phrases3,...
除了答案之外，你不能返回任何其他内容。
这些抽象意图词应满足以下要求：

抽象概念短语能够很好地代表该实体，并且可以是实体的类型或与实体相关的概念。
严格遵循提供的格式，不要添加多余的字符或词语。
如果可能的话，至少写出三个或更多不同抽象层次的短语。
不要在答案中重复相同的词语和输入的词语。
如果再也想不出更多的短语，立即停止，不需要解释。

实体：灵魂
背景：在BFI伦敦电影节首映，成为皮克斯票房最高的影片
你的答案：电影，影片

实体：Thinkpad X60
背景：理查德·斯托曼宣布他在Thinkpad X60上使用Trisquel
你的答案：Thinkpad，笔记本，机器，设备，硬件，电脑，品牌

实体：哈里·卡拉汉
背景：欺骗另一个抢劫犯，折磨天蝎座
你的答案：人，美国人，角色，警察，侦探

实体：黑山学院
背景：由约翰·安德鲁·赖斯创立，吸引了教师
你的答案：学院，大学，学校，文理学院

实体：4月1日
背景：乌特卡尔日庆祝
你的答案：日期，日子，时间，节日

实体：1991-4-19 
背景：中国船舶重工集团公司第七〇五研究所昆明分部, 注册日期
你的答案：日期，日子，时间，1991年，4月，19日

实体：2023年1月16日
背景：昆船智能技术股份有限公司于2023年1月16日召开第一届董事会第十九次会议和第一届监事会第八次会议
你的答案：2003年，1月，16日，日期，会议日期，日子，时间


实体：[ENTITY]
背景：[CONTEXT]
你的答案：
'''

RELATION_PROMPT_CN = '''我将给你一个关系。你需要给出几个包含1-2个词的短语来表示这个关系的抽象概念。
你必须按照以下格式返回你的答案：phrases1, phrases2, phrases3,...
除了答案之外，你不能返回任何其他内容。
这些抽象意图词应满足以下要求：

抽象关系短语能够很好地代表该关系，并且可以是关系的类型或与关系相关的概念。
严格遵循提供的格式，不要添加多余的字符或词语。
如果可能的话，至少写出三个或更多不同抽象层次的短语。
不要在答案中重复相同的词语和输入的词语。
如果再也想不出更多的短语，立即停止，不需要解释。
关系：参与
你的答案：成为一部分，参加，参与，投入，涉及

关系：被包括在内
你的答案：加入，成为一部分，成为成员，成为组成部分

关系：[RELATION]
你的答案：
'''
