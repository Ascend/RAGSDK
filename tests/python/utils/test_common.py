import os
import unittest

from loguru import logger
from mx_rag.utils.common import validate_sequence


class TestCommon(unittest.TestCase):
    def test_validate_str(self):
        # 字符串长度超过规格
        data = "a" * 1025
        self.assertFalse(validate_sequence(data))

    def test_validate_list(self):
        # 列表中元素长度超过规格
        data = ["a" * 1025]
        self.assertFalse(validate_sequence(data))

        # 列表长度超过规格
        data = ["a"] * 1025
        self.assertFalse(validate_sequence(data))

        data = [["a"]]
        self.assertFalse(validate_sequence(data))

    def test_validate_tuple(self):
        # 元组中元素长度超过规格
        data = ("a" * 1025,)
        self.assertFalse(validate_sequence(data))

        # 元组长度超过规格
        data = ("a",) * 1025
        self.assertFalse(validate_sequence(data))

    def test_validate_dict(self):
        # 字典中key长度超过规格
        data = {"a" * 1025: 1}
        self.assertFalse(validate_sequence(data))

        # 字典中value长度超过规格
        data = {"a": "b" * 1025}
        self.assertFalse(validate_sequence(data))

    def test_validate_dict_list(self):
        log_file = "./test.log"
        if os.path.exists(log_file):
            os.remove(log_file)
        logger.add(log_file, format="{message}", level="DEBUG")
        # 层数超过1错误
        validate_sequence({"a": ["b"]})
        # 第二层长度超过规格
        validate_sequence({"a": ["b"] * 1025}, max_check_depth=2)
        # 第三层长度超过规格错误
        validate_sequence({"a": [["b"] * 1025]}, max_check_depth=3)
        with open("./test.log") as fd:
            res1 = fd.readline()
            self.assertTrue(res1.find("nested depth cannot exceed 1") > -1)
            res2 = fd.readline()
            self.assertTrue(res2.find("1th layer param length") > -1)
            res3 = fd.readline()
            self.assertTrue(res3.find("2th layer param length") > -1)
            fd.close()
