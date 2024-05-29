# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import re

from loguru import logger

punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；，。&～、|\s:：\n ")
repeat_punc = set("!\"#$%&'()*,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——【】{};；，。&～、|\s:：\n ")


def is_alpha_or_digit_or_punc(char: str):
    if char.isalpha():
        return True
    elif char.isdigit():
        return True
    elif '\u4e00' <= char <= '\u9fff':  # 中文字符范围
        return True
    elif char in punctuation:
        return True
    else:
        logger.warning("remove char")
        return False


def remove_special_char(sentence: str):
    ans = ''
    for char in sentence:
        if is_alpha_or_digit_or_punc(char):
            ans += char
    return ans


def remove_duplicate_punctuation(sentence: str) -> str:
    """
    删除句子中重复的标点符号、重复的空格，同时将换行变为特殊字符'\n'
    """

    ans = ''
    n = len(sentence)
    p = 0
    while p < n:
        ans += sentence[p]
        while p + 1 < n and sentence[p] in repeat_punc and sentence[p + 1] in repeat_punc:
            p += 1
        p += 1

    return ans


def process_sentence(line: str) -> str:
    line = re.sub(r"\「|\」|\｢|\｣|\『|\』", '\"', line)  # 将「」｢｣『』这些符号替换成引号
    line = re.sub(r"\，\）|\；\）", '）', line)  # 罗德·法尼(Rod Dodji Fanni，）
    line = re.sub(r"\（\，|\(\，", '（', line)  # 阿魯拉·基馬(Alula Girma (，
    line = remove_special_char(line)
    line = remove_duplicate_punctuation(line)  # 删除中文空括号和重复的标点
    return line
