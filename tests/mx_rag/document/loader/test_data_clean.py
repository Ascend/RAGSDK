import unittest

from mx_rag.document.loader.data_clean import *


class DataCleanTestCase(unittest.TestCase):
    def test_is_alpha_or_digit_or_punc(self):
        self.assertEqual(is_alpha_or_digit_or_punc('a'), True)
        self.assertEqual(is_alpha_or_digit_or_punc('Z'), True)
        self.assertEqual(is_alpha_or_digit_or_punc('1'), True)
        self.assertEqual(is_alpha_or_digit_or_punc('用'), True)
        self.assertEqual(is_alpha_or_digit_or_punc('！'), True)

    def test_remove_special_char(self):
        self.assertEqual(remove_special_char('a'), "a")

    def test_remove_duplicate_punctuation(self):
        self.assertEqual(remove_duplicate_punctuation('a，，'), "a，")

    def test_process_sentence(self):
        self.assertEqual(process_sentence('a，，'), "a，")


if __name__ == '__main__':
    unittest.main()
