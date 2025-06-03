# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

JSON_REPAIR_PROMPT = '''
Repair the json text in {q}, the output structure should be strictly a list of dictionaries: [dict, dict, dict, ..., dict].
Ensure there is only one comma between each dictionary. Verify that the output has a complete square bracket enclosure and 
that no round brackets are present in the JSON text. The output should strictly contain the repaired JSON text, 
without any leading words or visible escape characters.
'''
