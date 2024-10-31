import unittest

from mx_rag.utils.common import validate_dict


class TestCommon(unittest.TestCase):
    def test_validate_dict_tuple(self):
        data = {"a": {"b": {"c": ("a" * 1025, "b" * 1025)}}}
        self.assertTrue(validate_dict(data))

    def test_validate_dict_set(self):
        data = {"a": {"b": {"c": {"a" * 1025}}}}
        self.assertTrue(validate_dict(data))

    def test_validate_dict_max_depth(self):
        data = {"a": {"b": {"c": {"d": {"1": "123"}}}}}
        self.assertFalse(validate_dict(data))

    def test_validate_dict_str_length(self):
        data = {"a": {"b": {"c": "key" * 1024 * 1024}}}
        self.assertFalse(validate_dict(data))
