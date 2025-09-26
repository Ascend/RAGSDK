#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from abc import ABC, abstractmethod


class PromptCompressor(ABC):
    @abstractmethod
    def compress_texts(self, context, question):
        pass


