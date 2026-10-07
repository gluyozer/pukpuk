import unittest

import numpy as np

from ideabot.nn import Adam, GRULanguageModel, sample_tokens


def make_batch(rng, V, B=3, T=6):
    tokens = rng.integers(0, V, size=(B, T + 1))
    mask = np.ones((B, T))
    mask[0, -2:] = 0  # часть позиций «дополнена» — как padding в обучении
    return tokens[:, :-1], tokens[:, 1:], mask


class GradientTest(unittest.TestCase):
    def test_backward_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        model = GRULanguageModel(vocab_size=11, emb_dim=5, hidden=7, seed=1, dtype=np.float64)
        # небольшие ненулевые смещения, чтобы проверить и градиенты по bias
        for name in ("b", "bo"):
            model.params[name] += rng.normal(0, 0.1, model.params[name].shape)
        inputs, targets, mask = make_batch(rng, 11)
        _, grads = model.loss_and_grads(inputs, targets, mask)

        eps = 1e-6
        checked = 0
        for name in GRULanguageModel.PARAM_NAMES:
            param = model.params[name]
            flat = param.reshape(-1)
            picks = rng.choice(flat.size, size=min(12, flat.size), replace=False)
            for idx in picks:
                old = flat[idx]
                flat[idx] = old + eps
                plus = model.evaluate(inputs, targets, mask)
                flat[idx] = old - eps
                minus = model.evaluate(inputs, targets, mask)
                flat[idx] = old
                numeric = (plus - minus) / (2 * eps)
                analytic = grads[name].reshape(-1)[idx]
                # относительный допуск + абсолютный (шум конечных разностей ~1e-10 при крошечных градиентах)
                tolerance = 1e-8 + 1e-4 * max(abs(numeric), abs(analytic))
                self.assertLess(abs(numeric - analytic), tolerance,
                                f"{name}[{idx}]: numeric={numeric:.3e} analytic={analytic:.3e}")
                checked += 1
        self.assertGreater(checked, 60)


class ModelBehaviourTest(unittest.TestCase):
    def test_step_matches_forward(self):
        """Пошаговая генерация должна давать те же logits, что и обучение."""
        rng = np.random.default_rng(3)
        model = GRULanguageModel(vocab_size=13, emb_dim=6, hidden=8, seed=2, dtype=np.float64)
        tokens = rng.integers(0, 13, size=(4, 9))
        logits, _, _ = model.forward(tokens)
        h = model.initial_state(4)
        for t in range(tokens.shape[1]):
            step_logits, h = model.step(tokens[:, t], h)
            np.testing.assert_allclose(step_logits, logits[:, t], atol=1e-10)

    def test_model_learns_a_pattern(self):
        rng = np.random.default_rng(5)
        V = 8
        model = GRULanguageModel(vocab_size=V, emb_dim=8, hidden=16, seed=4)
        optimizer = Adam(model.params, lr=1e-2)
        # детерминированная последовательность 1,2,3,4,5,6,1,2,...
        seq = np.array([[(i % 6) + 1 for i in range(13)] for _ in range(8)])
        inputs, targets, mask = seq[:, :-1], seq[:, 1:], np.ones((8, 12))
        first = model.evaluate(inputs, targets, mask)
        for _ in range(150):
            _, grads = model.loss_and_grads(inputs, targets, mask)
            optimizer.step(grads)
        last = model.evaluate(inputs, targets, mask)
        self.assertLess(last, first * 0.1)
        self.assertLess(last, 0.2)

    def test_sampling_respects_ban_and_greedy(self):
        rng = np.random.default_rng(0)
        logits = np.array([[0.0, 5.0, 1.0, 2.0]] * 50)
        picks = sample_tokens(logits, rng, temperature=1.0, top_p=1.0, banned=[1])
        self.assertNotIn(1, set(picks.tolist()))
        greedy = sample_tokens(logits, rng, temperature=0.0)
        self.assertTrue((greedy == 1).all())
        # top_p очень маленький -> всегда самый вероятный токен
        picks = sample_tokens(logits, rng, temperature=1.0, top_p=0.01)
        self.assertTrue((picks == 1).all())

    def test_save_and_load_roundtrip(self):
        import os
        import tempfile
        model = GRULanguageModel(vocab_size=9, emb_dim=4, hidden=5, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.npz")
            model.save(path, '["a", "b"]', extra={"note": "x"})
            loaded, vocab_json, config = GRULanguageModel.load(path)
        self.assertEqual(vocab_json, '["a", "b"]')
        self.assertEqual(config["note"], "x")
        for name in GRULanguageModel.PARAM_NAMES:
            np.testing.assert_array_equal(model.params[name], loaded.params[name])


if __name__ == "__main__":
    unittest.main()
