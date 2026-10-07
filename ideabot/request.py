"""Разбор запроса пользователя на естественном языке: вид проекта, темы, количество."""
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .lexicon import KIND_KEYS, THEMES

MAX_COUNT = 5
NUMBER_WORDS = {"один": 1, "одну": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5}
STOPWORDS = set("""
придумай придумайте придумать идею идеи идея идей идеями для про мне что нибудь что-нибудь давай хочу можешь
пожалуйста еще новую новые новый другую другие другой случайную случайные случайный проект проекта проекты
проектов проекту тему тема темы на или со по об как какой какая какое какие сделать сделай нужна нужно нужен
подскажи предложи дай покажи мой мою моё мои это там вот ещё есть быть чтобы свою свой свои игру игры
""".split())

_KIND_RES = {kind: [re.compile(p) for p in patterns] for kind, patterns in KIND_KEYS.items()}
_THEME_RES = {name: [re.compile(p) for p in data["keys"]] for name, data in THEMES.items()}


@dataclass
class Request:
    kind: Optional[str] = None
    themes: List[str] = field(default_factory=list)
    count: int = 1
    unknown: List[str] = field(default_factory=list)


def match_themes(word):
    """Все темы, ключи которых подходят слову (слово уже нормализовано)."""
    return [name for name, res in _THEME_RES.items() if any(r.fullmatch(word) for r in res)]


def normalize_words(text):
    return re.findall(r"[a-zа-я0-9]+(?:-[a-zа-я0-9]+)*", text.lower().replace("ё", "е"))


def parse_request(text, kind=None):
    """Достаёт из фразы вид проекта, до двух тем и число идей. kind — если вид задан командой."""
    request = Request(kind=kind)
    for word in normalize_words(text):
        if word.isdigit():
            if 1 <= int(word) <= MAX_COUNT and request.count == 1:
                request.count = int(word)
            continue
        if word in NUMBER_WORDS and request.count == 1:
            request.count = NUMBER_WORDS[word]
            continue
        matched_kind = next((k for k, res in _KIND_RES.items() if any(r.fullmatch(word) for r in res)), None)
        if matched_kind:
            request.kind = request.kind or matched_kind
            continue
        matched = match_themes(word)
        if matched:
            if matched[0] not in request.themes:
                request.themes.append(matched[0])
        elif len(word) >= 4 and word not in STOPWORDS:
            request.unknown.append(word)
    request.themes = request.themes[:2]
    return request
