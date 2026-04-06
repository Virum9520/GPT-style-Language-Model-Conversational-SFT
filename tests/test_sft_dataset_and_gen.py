import os
import unittest
import torch

from gpt import setup_tokenizer, GPTModel
import sft
from sft_gpt import setup_tokenizer as sft_setup_tokenizer


def _find_jsonl():
    cands = [
        "Data/smol-smoltalk-dev.jsonl.gz",
        "Data/smol-smoltalk-train.jsonl.gz",
        "Data/sft_data_packed.arrow/smol-smoltalk-dev.jsonl.gz",  # fallback if user reorganized
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find a smol-smoltalk jsonl(.gz) file under Data/.")


class TestSFTDatasetAndCollator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jsonl_path = _find_jsonl()
        cls.tokenizer = setup_tokenizer()

    def test_dataset_and_collator(self):
        ds = sft.SFTDataset(self.jsonl_path, self.tokenizer, max_length=128)
        self.assertGreater(len(ds), 0)
        ids, labs = ds[0]
        self.assertEqual(ids.shape, labs.shape)
        # there should be at least some trainable labels (assistant tokens)
        self.assertGreater(int((labs != -100).sum().item()), 0)

        from torch.utils.data import DataLoader
        dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=sft.sft_data_collator)
        batch = next(iter(dl))
        self.assertIn("input_ids", batch)
        self.assertIn("labels", batch)
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        # padded labels should be -100
        self.assertTrue(((batch["labels"] == -100) | (batch["labels"] >= 0)).all())

    def test_generate_chat_response_shape(self):
        tok, vocab = sft_setup_tokenizer()
        cfg = {
            "vocab_size": vocab,
            "context_length": 128,
            "emb_dim": 128,
            "n_heads": 4,
            "n_layers": 2,
            "drop_rate": 0.1,
            "qkv_bias": False,
        }
        model = GPTModel(cfg)
        model.eval()
        # Just ensure it runs and returns a string
        txt = sft.generate_chat_response(model, tok, "Hello!")
        self.assertIsInstance(txt, str)


if __name__ == "__main__":
    unittest.main()


