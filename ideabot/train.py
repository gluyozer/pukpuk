"""Обучение нейросети. Каждую эпоху корпус генерируется заново — переобучиться нельзя."""
import math
import time
from pathlib import Path

import numpy as np

from . import corpus
from .nn import Adam, GRULanguageModel
from .tokenizer import Vocab

MODEL_PATH = Path(__file__).resolve().parent / "model" / "ideabot.npz"

# настройки, с которыми обучена поставляемая модель (используются и командой train, и автообучением)
DEFAULT_EPOCHS = 12
DEFAULT_SEQS_PER_EPOCH = 12000
DEFAULT_HIDDEN = 192


def make_batches(samples, vocab, batch_size, rng):
    """Склеивает идеи в батчи (похожей длины — меньше padding). Возвращает (inputs, targets, mask)."""
    encoded = [vocab.encode(s.tokens()) for s in samples]
    order = np.argsort([len(e) + rng.random() for e in encoded])  # сортировка по длине с шумом
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    for idx in batches:
        seqs = [encoded[i] for i in idx]
        width = max(len(s) for s in seqs)
        tokens = np.full((len(seqs), width), vocab.pad_id, dtype=np.int64)
        lengths = np.array([len(s) for s in seqs])
        for row, seq in enumerate(seqs):
            tokens[row, :len(seq)] = seq
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        position = np.arange(inputs.shape[1])[None, :]
        # учим предсказывать только текст идеи: первые PREFIX_LEN токенов (вид, темы) нам задаёт пользователь
        mask = ((position + 1 >= corpus.PREFIX_LEN) & (position + 1 < lengths[:, None])).astype(np.float32)
        yield inputs, targets, mask


def evaluate(model, vocab, samples, batch_size=128):
    total, count = 0.0, 0.0
    rng = np.random.default_rng(0)
    for inputs, targets, mask in make_batches(samples, vocab, batch_size, rng):
        total += model.evaluate(inputs, targets, mask) * float(mask.sum())
        count += float(mask.sum())
    return float(total / max(count, 1.0))


def train(epochs=DEFAULT_EPOCHS, seqs_per_epoch=DEFAULT_SEQS_PER_EPOCH, batch_size=64, hidden=DEFAULT_HIDDEN,
          emb_dim=64, lr=4e-3, seed=0,
          out_path=MODEL_PATH, log=print):
    vocab = Vocab.build(corpus.vocabulary_texts())
    model = GRULanguageModel(len(vocab), emb_dim=emb_dim, hidden=hidden, seed=seed)
    optimizer = Adam(model.params, lr=lr)
    rng = np.random.default_rng(seed)
    held_out = corpus.make_corpus(1000, seed=10_000_019)
    steps_per_epoch = math.ceil(seqs_per_epoch / batch_size)
    total_steps, step = epochs * steps_per_epoch, 0
    log(f"словарь: {len(vocab)} токенов, параметров: {model.num_params():,}, шагов: {total_steps}")
    started = time.time()
    for epoch in range(1, epochs + 1):
        samples = corpus.make_corpus(seqs_per_epoch, seed=seed * 1000 + epoch)
        running, batches = 0.0, 0
        for inputs, targets, mask in make_batches(samples, vocab, batch_size, rng):
            warmup = min(1.0, (step + 1) / 30)
            cosine = 0.5 * (1 + math.cos(math.pi * step / total_steps))
            loss, grads = model.loss_and_grads(inputs, targets, mask)
            optimizer.step(grads, lr=lr * warmup * (0.05 + 0.95 * cosine))
            running += loss
            batches += 1
            step += 1
        val = evaluate(model, vocab, held_out)
        log(f"эпоха {epoch}/{epochs}  loss {running / batches:.4f}  val {val:.4f}  "
            f"perplexity {math.exp(val):.2f}  ({time.time() - started:.0f} с)")
        save(model, vocab, out_path, epoch=epoch, val_loss=val)
    return model, vocab


def save(model, vocab, path, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path, vocab.to_json(), extra=extra)
