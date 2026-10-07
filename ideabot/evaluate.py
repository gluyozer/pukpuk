"""Оценка качества обученной модели: доля валидных идей, следование теме, разнообразие."""
import random
import time

from . import corpus
from . import lexicon as L
from .generate import IdeaGenerator, check_idea
from .request import normalize_words


def theme_words(theme):
    """Слова из словаря темы (основы длиной 5+ символов) — по ним смотрим, что идея «про это»."""
    data = L.THEMES[theme]
    words = set()
    for key in ("tails", "heroes", "places", "goals", "threats", "about", "fans"):
        for phrase in data[key]:
            words.update(w[:5] for w in normalize_words(phrase) if len(w) >= 5)
    words.update(w[:5] for w, _ in data["nouns"] if len(w) >= 5)
    return words


def evaluate(generator: IdeaGenerator, per_kind=40, per_theme=12, log=print, show=6):
    report = {}
    started = time.time()

    total = valid = 0
    reasons = {}
    for kind in L.KINDS:
        for cand in generator.sample_candidates(per_kind, kind=kind):
            total += 1
            ok, why = (check_idea(cand.kind, cand.tokens) if cand.finished else (False, "не закончилась"))
            if ok:
                valid += 1
            else:
                reasons[why] = reasons.get(why, 0) + 1
    report["valid_rate"] = valid / total
    log(f"валидных идей: {valid}/{total} = {valid / total:.1%}"
        + (f"  (брак: {reasons})" if reasons else ""))

    hits = checked = 0
    for theme in corpus.THEME_NAMES:
        words = theme_words(theme)
        for idea in generator.generate(n=per_theme, themes=[theme]):
            checked += 1
            text_words = {w[:5] for w in normalize_words(idea.one_line())}
            hits += bool(words & text_words)
    report["theme_rate"] = hits / checked
    log(f"идей по заданной теме: {hits}/{checked} = {hits / checked:.1%}")

    ideas = generator.generate(n=200, max_rounds=30)
    titles = {i.title.lower() for i in ideas}
    texts = {i.one_line() for i in ideas}
    report["unique_rate"] = len(texts) / max(len(ideas), 1)
    log(f"уникальных идей в выборке из {len(ideas)}: {len(texts)}, уникальных названий: {len(titles)}")

    reference = {s.text for s in corpus.make_corpus(30000, seed=777)}
    copies = sum(i.one_line().lower() in reference for i in ideas)
    report["copy_rate"] = copies / max(len(ideas), 1)
    log(f"идей, дословно совпавших с обучающей выборкой: {copies}/{len(ideas)}")

    log(f"время оценки: {time.time() - started:.0f} с\n\nпримеры:")
    for idea in random.Random(0).sample(ideas, min(show, len(ideas))):
        log(idea.format() + "\n")
    return report
