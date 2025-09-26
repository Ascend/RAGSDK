#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import unittest

from mx_rag.tools.finetune.instruction.rule_driven_complex_instruction import RuleComplexInstructionRewriter


class TestRuleDrivenComplexInstruction(unittest.TestCase):
    def test_run_success(self):
        rewriter = RuleComplexInstructionRewriter()
        rewriter.get_rewrite_prompts('求客房部主管年终总结及来年工作计划？', '更改指令语言风格')


if __name__ == '__main__':
    unittest.main()
