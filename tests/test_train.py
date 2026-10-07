import tempfile
import unittest
from pathlib import Path

from ideabot import train
from ideabot.generate import IdeaGenerator


class TrainingPipelineTest(unittest.TestCase):
    def test_tiny_training_run_end_to_end(self):
        """Маленькая модель с нуля: обучение уменьшает ошибку, файл сохраняется и грузится, генерация работает."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.npz"
            logs = []
            model, vocab = train.train(epochs=2, seqs_per_epoch=256, batch_size=32, hidden=24, emb_dim=12,
                                       out_path=path, log=logs.append)
            self.assertTrue(path.exists())
            losses = [float(line.split("val ")[1].split()[0]) for line in logs if "val " in line]
            self.assertEqual(len(losses), 2)
            self.assertLess(losses[1], losses[0])
            generator = IdeaGenerator.load(path, seed=0)
            candidates = generator.sample_candidates(4, kind="игра", themes=["космос"])
            self.assertEqual(len(candidates), 4)
            self.assertTrue(all(c.tokens for c in candidates))
            self.assertEqual(generator.vocab.tokens, vocab.tokens)

    def test_prefix_is_not_trained_on(self):
        import numpy as np
        from ideabot import corpus
        samples = corpus.make_corpus(16, seed=1)
        from ideabot.tokenizer import Vocab
        vocab = Vocab.build(corpus.vocabulary_texts())
        inputs, targets, mask = next(train.make_batches(samples, vocab, 16, np.random.default_rng(0)))
        # первые два предсказания (вид -> тема1, тема1 -> тема2) и третье (тема2 -> «) — маска: считаем только с третьего входа
        self.assertTrue((mask[:, :2] == 0).all())
        self.assertTrue((mask[:, 2] == 1).all())
        self.assertTrue((targets[mask == 1] != vocab.pad_id).all())


if __name__ == "__main__":
    unittest.main()
