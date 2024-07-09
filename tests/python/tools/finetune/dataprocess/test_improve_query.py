# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import unittest

from mx_rag.tools.finetune.dataprocess.improve_query import RuleComplexInstructionRewriter


class TestImproveQuery(unittest.TestCase):

    def test_run_success(self):
        rewriter = RuleComplexInstructionRewriter()
        self.assertNotEqual(rewriter.get_rewrite_prompts("请问你是谁", "增加指令子任务"), "")
        self.assertNotEqual(rewriter.get_rewrite_prompts("请问你是谁", "增加回答的限制"), "")
        self.assertNotEqual(rewriter.get_rewrite_prompts("请问你是谁", "增加领域知识"), "")
        self.assertNotEqual(rewriter.get_rewrite_prompts("请问你是谁", "增加指令格式"), "")
        self.assertNotEqual(rewriter.get_rewrite_prompts("请问你是谁", "增加指令要求"), "")
        self.assertNotEqual(rewriter.get_rewrite_prompts("请问你是谁", "更改指令语言风格"), "")

    def test_run_failed(self):
        rewriter = RuleComplexInstructionRewriter()
        self.assertEqual(rewriter.get_rewrite_prompts("请问你是谁", "不存在的指令"), "")


if __name__ == '__main__':
    unittest.main()
