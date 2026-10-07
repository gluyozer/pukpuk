"""Генерация идей нейросетью: сэмплирование, проверка структуры, отбор."""
import random
from dataclasses import dataclass

import numpy as np

from . import corpus
from . import lexicon as L
from .idea import Idea
from .nn import GRULanguageModel, sample_tokens
from .tokenizer import Vocab, detokenize

MAX_BODY_TOKENS = 90
DEFAULT_TEMPERATURE = 0.85
DEFAULT_TOP_P = 0.92


def check_idea(kind, tokens):
    """Проверяет, что текст — цельная идея нужного вида из фраз словаря с правильным названием.
    Возвращает (ok, причина). Так отсеиваются «склейки» нейросети с ошибками согласования."""
    match, reason = corpus.parse_idea(kind, detokenize(tokens, pretty=False))
    return match is not None, reason


@dataclass
class Candidate:
    kind: str
    themes: tuple
    tokens: list
    finished: bool


class IdeaGenerator:
    def __init__(self, model, vocab, seed=None):
        self.model = model
        self.vocab = vocab
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.pyrng = random.Random(seed)
        self._banned = np.array(vocab.forbidden_ids, dtype=np.int64)

    @classmethod
    def load(cls, path=None, seed=None):
        from .train import MODEL_PATH
        model, vocab_json, _ = GRULanguageModel.load(path or MODEL_PATH)
        return cls(model, Vocab.from_json(vocab_json), seed=seed)

    # ---------------------------------------------------------------- выбор вида и тем
    def _prefix(self, kind, themes):
        kind = kind or self.pyrng.choice(L.KINDS)
        if themes:
            themes = tuple(themes[:2])
        else:
            themes = [self.pyrng.choice(corpus.THEME_NAMES)]
            if self.pyrng.random() < corpus.MASHUP_PROB:
                themes.append(self.pyrng.choice([t for t in corpus.THEME_NAMES if t != themes[0]]))
            themes = tuple(themes)
        return kind, themes

    # ---------------------------------------------------------------- сэмплирование
    def sample_candidates(self, count, kind=None, themes=None, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P):
        """Генерирует count текстов параллельно (одним батчем). Каждому задаём свой вид и темы."""
        prefixes = [self._prefix(kind, themes) for _ in range(count)]
        ids = np.array([self.vocab.encode(corpus.prefix_tokens(k, t)) for k, t in prefixes])
        model = self.model
        h = model.initial_state(count)
        for position in range(corpus.PREFIX_LEN):
            logits, h = model.step(ids[:, position], h)
        bodies = [[] for _ in range(count)]
        alive = np.ones(count, dtype=bool)
        for _ in range(MAX_BODY_TOKENS):
            nxt = sample_tokens(logits, self.rng, temperature, top_p, banned=self._banned)
            for i in np.flatnonzero(alive):
                if nxt[i] == self.vocab.end_id:
                    alive[i] = False
                else:
                    bodies[i].append(int(nxt[i]))
            if not alive.any():
                break
            logits, h = model.step(nxt, h)
        return [Candidate(k, t, self.vocab.decode(body), finished=not alive[i])
                for i, ((k, t), body) in enumerate(zip(prefixes, bodies))]

    # ---------------------------------------------------------------- публичный API
    def generate(self, n=1, kind=None, themes=None, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P,
                 exclude=(), max_rounds=8):
        """Возвращает до n уникальных идей, прошедших проверку структуры."""
        seen = {t.lower() for t in exclude}
        ideas = []
        for _ in range(max_rounds):
            for cand in self.sample_candidates(max(8, 4 * (n - len(ideas))), kind, themes, temperature, top_p):
                if not cand.finished or not check_idea(cand.kind, cand.tokens)[0]:
                    continue
                idea = Idea.from_tokens(cand.kind, cand.themes, cand.tokens)
                if idea.title.lower() in seen:
                    continue
                seen.add(idea.title.lower())
                ideas.append(idea)
                if len(ideas) == n:
                    return ideas
        return ideas
