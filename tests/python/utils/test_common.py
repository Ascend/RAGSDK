import unittest

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
        # 层数超过1错误
        data = {"a": ["b"]}
        self.assertFalse(validate_sequence(data))

        # 第二层长度超过规格
        data = {"a": ["b"] * 1025}
        self.assertFalse(validate_sequence(data, max_check_depth=2))

        # 第三层长度超过规格错误
        data = {"a": [["b"] * 1025]}
        self.assertFalse(validate_sequence(data, max_check_depth=3))
