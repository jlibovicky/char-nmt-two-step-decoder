import unittest

import numpy as np
import torch

from bigram_tokenizer import BigramTokenizer, from_data, SPECIAL_SYMBOLS


class TestBigramTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = from_data(["ABC", "ABCD"])

    def test_from_data_1(self):
        data = ["A", "B", "C"]
        tokenizer = from_data(data)

        self.assertEqual(tokenizer.idx_to_str[0], "<pad>")
        self.assertEqual(tokenizer.idx_to_str[1], "<s>")
        self.assertEqual(tokenizer.idx_to_str[2], "</s>")
        self.assertEqual(tokenizer.idx_to_str[3], "<unk>")

        self.assertIn("A", tokenizer.str_to_idx)
        self.assertIn("B", tokenizer.str_to_idx)
        self.assertIn("C", tokenizer.str_to_idx)
        self.assertIn("<s>A", tokenizer.str_to_idx)
        self.assertIn("A</s>", tokenizer.str_to_idx)
        self.assertIn("<s>B", tokenizer.str_to_idx)
        self.assertIn("B</s>", tokenizer.str_to_idx)
        self.assertIn("<s>C", tokenizer.str_to_idx)
        self.assertIn("C</s>", tokenizer.str_to_idx)

        self.assertEqual(tokenizer.vocab_size, 13)

    def test_from_data_limited_data(self):
        data = ["A", "B", "C"]
        tokenizer = from_data(data, max_lines=2)

        self.assertEqual(tokenizer.idx_to_str[0], "<pad>")
        self.assertEqual(tokenizer.idx_to_str[1], "<s>")
        self.assertEqual(tokenizer.idx_to_str[2], "</s>")
        self.assertEqual(tokenizer.idx_to_str[3], "<unk>")
        self.assertNotIn("C", tokenizer.idx_to_str)

    def test_batch_encode_plus_pt(self):
        encoded = self.tokenizer.batch_encode_plus(
            ["ABC", "ABCD"], return_tensors="pt")
        self.assertIsInstance(encoded, tuple)
        self.assertIsInstance(encoded[0], torch.Tensor)

    def test_encode_decode(self):
        encoded = self.tokenizer.batch_encode_plus(
            ["ABC", "ABCD"], return_tensors="pt")
        decoded = self.tokenizer.batch_decode(encoded[0])

        self.assertEqual(decoded[0], "ABC")
        self.assertEqual(decoded[1], "ABCD")


if __name__ == '__main__':
    unittest.main()
