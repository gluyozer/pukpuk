"""Нейросеть на чистом numpy: языковая модель на GRU с ручным обратным проходом (BPTT).

Архитектура:  токен -> эмбеддинг -> GRU -> линейный слой -> softmax по словарю.
GRU:  z = σ(xWz + hUz),  r = σ(xWr + hUr),  n = tanh(xWn + (r*h)Un),  h' = (1-z)*n + z*h
"""
import json

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class GRULanguageModel:
    PARAM_NAMES = ("E", "Wx", "Wh_zr", "Wh_n", "b", "Wo", "bo")

    def __init__(self, vocab_size, emb_dim=64, hidden=160, seed=0, dtype=np.float32):
        self.V, self.D, self.H = vocab_size, emb_dim, hidden
        self.dtype = dtype
        rng = np.random.default_rng(seed)

        def uniform(shape, fan_in):
            limit = 1.0 / np.sqrt(fan_in)
            return rng.uniform(-limit, limit, shape).astype(dtype)

        self.params = {
            "E": (rng.standard_normal((vocab_size, emb_dim)) * 0.1).astype(dtype),
            "Wx": uniform((emb_dim, 3 * hidden), emb_dim),
            "Wh_zr": uniform((hidden, 2 * hidden), hidden),
            "Wh_n": uniform((hidden, hidden), hidden),
            "b": np.zeros(3 * hidden, dtype),
            "Wo": uniform((hidden, vocab_size), hidden),
            "bo": np.zeros(vocab_size, dtype),
        }

    # ---------------------------------------------------------------- прямой проход
    def _cell(self, xw, h):
        """Один шаг GRU. xw — уже посчитанный x@Wx+b, форма [B, 3H]."""
        H = self.H
        p = self.params
        zr = sigmoid(xw[:, :2 * H] + h @ p["Wh_zr"])
        z, r = zr[:, :H], zr[:, H:]
        n = np.tanh(xw[:, 2 * H:] + (r * h) @ p["Wh_n"])
        h_new = (1.0 - z) * n + z * h
        return h_new, z, r, n

    def forward(self, tokens, h0=None):
        """tokens: int [B, T]. Возвращает logits [B, T, V], кэш для backward и последнее состояние."""
        p = self.params
        B, T = tokens.shape
        x = p["E"][tokens]  # [B, T, D]
        xw = x @ p["Wx"] + p["b"]  # [B, T, 3H]
        hs = np.zeros((B, T + 1, self.H), self.dtype)
        if h0 is not None:
            hs[:, 0] = h0
        zs = np.zeros((B, T, self.H), self.dtype)
        rs = np.zeros_like(zs)
        ns = np.zeros_like(zs)
        for t in range(T):
            hs[:, t + 1], zs[:, t], rs[:, t], ns[:, t] = self._cell(xw[:, t], hs[:, t])
        logits = hs[:, 1:] @ p["Wo"] + p["bo"]
        cache = (tokens, x, hs, zs, rs, ns)
        return logits, cache, hs[:, -1]

    def step(self, tokens, h):
        """Один шаг для генерации: tokens [B], h [B, H] -> logits [B, V], новое h."""
        p = self.params
        xw = p["E"][tokens] @ p["Wx"] + p["b"]
        h_new, *_ = self._cell(xw, h)
        return h_new @ p["Wo"] + p["bo"], h_new

    def initial_state(self, batch):
        return np.zeros((batch, self.H), self.dtype)

    # ---------------------------------------------------------------- потери и градиенты
    def loss_and_grads(self, inputs, targets, mask):
        """Средняя кросс-энтропия по токенам с mask=1 и градиенты по всем параметрам."""
        p = self.params
        logits, cache, _ = self.forward(inputs)
        tokens, x, hs, zs, rs, ns = cache
        B, T = inputs.shape
        H = self.H

        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        count = max(float(mask.sum()), 1.0)
        rows, cols = np.indices((B, T))
        picked = probs[rows, cols, targets]
        loss = float(-(np.log(picked + 1e-12) * mask).sum() / count)

        dlogits = probs
        dlogits[rows, cols, targets] -= 1.0
        dlogits *= (mask / count)[..., None].astype(self.dtype)

        grads = {name: np.zeros_like(value) for name, value in p.items()}
        hs_out = hs[:, 1:]
        grads["Wo"] = hs_out.reshape(-1, H).T @ dlogits.reshape(-1, self.V)
        grads["bo"] = dlogits.sum(axis=(0, 1))
        dhs = dlogits @ p["Wo"].T  # [B, T, H]

        dxw = np.zeros((B, T, 3 * H), self.dtype)
        dh_next = np.zeros((B, H), self.dtype)
        for t in reversed(range(T)):
            dh = dhs[:, t] + dh_next
            h_prev, z, r, n = hs[:, t], zs[:, t], rs[:, t], ns[:, t]
            dn_pre = dh * (1.0 - z) * (1.0 - n * n)
            dz_pre = dh * (h_prev - n) * z * (1.0 - z)
            drh = dn_pre @ p["Wh_n"].T
            dr_pre = drh * h_prev * r * (1.0 - r)
            dzr = np.concatenate([dz_pre, dr_pre], axis=1)
            grads["Wh_n"] += (r * h_prev).T @ dn_pre
            grads["Wh_zr"] += h_prev.T @ dzr
            dh_next = dh * z + drh * r + dzr @ p["Wh_zr"].T
            dxw[:, t, :2 * H] = dzr
            dxw[:, t, 2 * H:] = dn_pre

        grads["Wx"] = x.reshape(-1, self.D).T @ dxw.reshape(-1, 3 * H)
        grads["b"] = dxw.sum(axis=(0, 1))
        dx = dxw @ p["Wx"].T
        np.add.at(grads["E"], tokens, dx)
        return loss, grads

    def evaluate(self, inputs, targets, mask):
        logits, _, _ = self.forward(inputs)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        rows, cols = np.indices(inputs.shape)
        return float(-(logp[rows, cols, targets] * mask).sum() / max(float(mask.sum()), 1.0))

    def num_params(self):
        return sum(v.size for v in self.params.values())

    # ---------------------------------------------------------------- сохранение
    def save(self, path, vocab_json, extra=None):
        config = {"V": self.V, "D": self.D, "H": self.H, **(extra or {})}
        np.savez_compressed(path, config=np.array(json.dumps(config)), vocab=np.array(vocab_json), **self.params)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            config = json.loads(str(data["config"]))
            model = cls(config["V"], config["D"], config["H"], dtype=np.float32)
            for name in cls.PARAM_NAMES:
                model.params[name] = data[name].astype(np.float32)
            return model, str(data["vocab"]), config


class Adam:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-8, clip=5.0):
        self.params, self.lr, self.betas, self.eps, self.clip = params, lr, betas, eps, clip
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads, lr=None):
        lr = self.lr if lr is None else lr
        norm = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values())))
        scale = min(1.0, self.clip / (norm + 1e-8))
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for name, g in grads.items():
            g = g * scale
            self.m[name] = b1 * self.m[name] + (1 - b1) * g
            self.v[name] = b2 * self.v[name] + (1 - b2) * g * g
            self.params[name] -= (lr * (self.m[name] / c1) / (np.sqrt(self.v[name] / c2) + self.eps)).astype(
                self.params[name].dtype)
        return norm


def sample_tokens(logits, rng, temperature=0.9, top_p=0.95, banned=None, bias=None):
    """Выбирает следующий токен для каждой строки logits [K, V] (температура + nucleus-сэмплинг)."""
    logits = logits.astype(np.float64)
    if bias is not None:
        logits = logits + bias
    if banned is not None and len(banned):
        logits[:, banned] = -np.inf
    if temperature <= 1e-6:
        return logits.argmax(axis=1)
    logits = logits / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    keep = np.cumsum(sorted_p, axis=1) - sorted_p < top_p  # всегда оставляем хотя бы самый вероятный
    sorted_p = np.where(keep, sorted_p, 0.0)
    sorted_p /= sorted_p.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(sorted_p, axis=1)
    pick = (rng.random((len(probs), 1)) > cumulative).sum(axis=1)
    pick = np.minimum(pick, probs.shape[1] - 1)
    return order[np.arange(len(probs)), pick]
