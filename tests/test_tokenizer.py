import unittest

import numpy as np
import torch


from char_tokenizer import CharTokenizer, from_data, SPECIAL_SYMBOLS


class TestCharTokenizer(unittest.TestCase):

    def test_from_data_1(self):
        data = ["A", "B", "C"]
        tokenizer = from_data(data)

        self.assertEqual(tokenizer.idx_to_str[0], "<pad>")
        self.assertEqual(tokenizer.idx_to_str[1], "<s>")
        self.assertEqual(tokenizer.idx_to_str[2], "</s>")
        self.assertEqual(tokenizer.idx_to_str[3], "<unk>")
        self.assertEqual(tokenizer.idx_to_str[4], "A")
        self.assertEqual(tokenizer.idx_to_str[5], "B")
        self.assertEqual(tokenizer.idx_to_str[6], "C")

    def test_from_data_limited_data(self):
        data = ["A", "B", "C"]
        tokenizer = from_data(data, max_lines=2)

        self.assertEqual(tokenizer.idx_to_str[0], "<pad>")
        self.assertEqual(tokenizer.idx_to_str[1], "<s>")
        self.assertEqual(tokenizer.idx_to_str[2], "</s>")
        self.assertEqual(tokenizer.idx_to_str[3], "<unk>")
        self.assertNotIn("C", tokenizer.idx_to_str)

    def setUp(self):
        self.tokenizer = CharTokenizer(SPECIAL_SYMBOLS + list("ABCD"))
        self.abc_abcd_output = [[1, 4, 5, 6, 2, 0], [1, 4, 5, 6, 7, 2]]

    def test_vocab_size(self):
        self.assertEqual(self.tokenizer.vocab_size, 8)

    def test_batch_encode_plus_np(self):
        encoded = self.tokenizer.batch_encode_plus(
            ["ABC", "ABCD"], return_tensors="np")
        self.assertIsInstance(encoded, tuple)
        self.assertIsInstance(encoded[0], np.ndarray)

        self.assertTrue(
            np.all(encoded[0] == np.array(self.abc_abcd_output)))
        self.assertEqual(encoded[1].sum(), 11)

    def test_batch_encode_plus_pt(self):
        encoded = self.tokenizer.batch_encode_plus(
            ["ABC", "ABCD"], return_tensors="pt")
        self.assertIsInstance(encoded, tuple)
        self.assertIsInstance(encoded[0], torch.Tensor)

        # pylint: disable=not-callable
        self.assertTrue(
            torch.all(encoded[0] == torch.tensor(self.abc_abcd_output)))
        # pylint: enable=not-callable
        self.assertEqual(encoded[1].sum(), 11)

    def test_batch_encode_plus_not_pad(self):
        encoded = self.tokenizer.batch_encode_plus(
            ["ABC", "ABCD"],
            pad_to_max_length=False,
            return_attention_mask=False)
        self.assertEqual(
            encoded[0].numpy().tolist(), self.abc_abcd_output[0][:-1])
        self.assertEqual(encoded[1].numpy().tolist(), self.abc_abcd_output[1])

    def test_batch_encode_plus_types(self):
        encoded = self.tokenizer.batch_encode_plus(
            ["ABC", "ABCD"],
            return_attention_mask=True)
        self.assertEqual(encoded[0].dtype, torch.int64)
        self.assertEqual(encoded[1].dtype, torch.float)

    def test_decode(self):
        decoded = self.tokenizer.batch_decode(self.abc_abcd_output)
        self.assertEqual(decoded, ["ABC", "ABCD"])

    def test_decode_np(self):
        decoded = self.tokenizer.batch_decode(np.array(self.abc_abcd_output))
        self.assertEqual(decoded, ["ABC", "ABCD"])

    def test_decode_pt(self):
        # pylint: disable=not-callable
        decoded = self.tokenizer.batch_decode(
            torch.tensor(self.abc_abcd_output))
        # pylint: enable=not-callable
        self.assertEqual(decoded, ["ABC", "ABCD"])


if __name__ == '__main__':
    unittest.main()
