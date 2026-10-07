"""Генератор обучающего корпуса.

Идея — это последовательность токенов:  <вид> <тема1> <тема2|нет> « название » — текст ... <end>
Нейросеть учится продолжать такие последовательности, а при генерации мы сами
задаём первые три токена (вид и темы) — так пользователь управляет результатом.
"""
import random
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from . import lexicon as L
from .tokenizer import END, NONE, kind_token, theme_token, tokenize

PREFIX_LEN = 3  # вид, тема 1, тема 2
MASHUP_PROB = 0.35  # доля идей на стыке двух тем
THEME_NAMES = list(L.THEMES)


@dataclass(frozen=True)
class Sample:
    kind: str
    themes: Tuple[str, ...]  # одна или две темы
    text: str  # строчный текст идеи без служебных токенов

    def tokens(self):
        second = theme_token(self.themes[1]) if len(self.themes) > 1 else NONE
        return [kind_token(self.kind), theme_token(self.themes[0]), second] + tokenize(self.text) + [END]


def prefix_tokens(kind, themes):
    second = theme_token(themes[1]) if len(themes) > 1 else NONE
    return [kind_token(kind), theme_token(themes[0]), second]


def agree(stem, adj_type, gender):
    """Согласует прилагательное с родом существительного: ('тих','velar','f') -> 'тихая'."""
    m, f, n = L.ADJ_ENDINGS[adj_type]
    return stem + {"m": m, "f": f, "n": n}[gender]


def all_adjective_forms():
    for adj_type, stems in L.ADJECTIVES.items():
        for stem in stems:
            for gender in "mfn":
                yield agree(stem, adj_type, gender)


class _Picker:
    """Выбирает слова из словаря одной или двух тем (для «стыка» тем — случайно из любой)."""

    def __init__(self, rng, themes):
        self.rng = rng
        self.themes = themes

    def get(self, key):
        theme = self.rng.choice(self.themes)
        return self.rng.choice(L.THEMES[theme][key])

    def get_other(self, key, taken):
        for _ in range(20):
            value = self.get(key)
            if value != taken:
                return value
        return value


def make_title(rng, picker):
    roll = rng.random()
    if roll < 0.45:
        theme = rng.choice(picker.themes)
        noun, gender = rng.choice(L.THEMES[theme]["nouns"])
        adj_type = rng.choice(list(L.ADJECTIVES))
        return f"{agree(rng.choice(L.ADJECTIVES[adj_type]), adj_type, gender)} {noun}"
    if roll < 0.75:
        theme = rng.choice(picker.themes)
        noun = rng.choice(L.THEMES[theme]["nouns"])[0]
        tail = picker.get("tails")
        for _ in range(10):  # без повторов вроде «Пицца Волшебной Пиццы»
            if noun[:4] not in tail:
                break
            tail = picker.get("tails")
        return f"{noun} {tail}"
    if roll < 0.85:
        theme = rng.choice(picker.themes)
        first, second = rng.sample(L.THEMES[theme]["nouns"], 2)
        return f"{first[0]} и {second[0]}"
    theme = rng.choice(picker.themes)
    return rng.choice(L.THEMES[theme]["nouns"])[0]


# --- Шаблоны ------------------------------------------------------------------
# Единственный источник истины: из этой таблицы строятся и обучающие идеи, и проверка (check_idea).
# {title} — название, {hero2} — второй герой (не совпадает с первым); остальные слоты ниже.
TEMPLATES = {
    "игра": [
        "«{title}» — {genre}. ты — {hero} {place}, и тебе нужно {goal}, пока {threat}. фишка: {mech}.",
        "«{title}» — {genre} про {about}. ты — {hero}, и тебе нужно {goal} {place}. фишка: {mech}.",
        "«{title}» — {coop} {place}: один игрок — {hero}, другой — {hero2}. "
        "вместе им нужно {goal}, пока {threat}. фишка: {mech}.",
        "«{title}» — {genre} про {about}. главный герой — {hero} {place}. время поджимает: {threat}. фишка: {mech}.",
    ],
    "приложение": [
        "«{title}» — мобильное приложение про {about}, которое {app_benefit}. "
        "для кого: для любителей {fans}. фишка: {app_feature}.",
    ],
    "бот": [
        "«{title}» — telegram-бот про {about}, который {bot_benefit}. "
        "для кого: для любителей {fans}. фишка: {bot_feature}.",
    ],
    "сайт": [
        "«{title}» — веб-сервис про {about}, где {web_does}. для кого: для любителей {fans}. фишка: {web_feature}.",
    ],
    "ии": [
        "«{title}» — ии-проект: нейросеть, которая {ai_verb} про {about}. "
        "для кого: для любителей {fans}. фишка: {ai_feature}.",
    ],
    "железо": [
        "«{title}» — самодельное устройство на {board}, которое {hw_verb} {about}. "
        "для кого: для любителей {fans}. фишка: {hw_feature}.",
    ],
    "творчество": [
        "«{title}» — творческий проект: {form} про {about}. для кого: для любителей {fans}. фишка: {twist}.",
    ],
}

# слоты, общие для всех тем
GLOBAL_SLOTS = {
    "genre": L.GENRES, "coop": L.COOP_GENRES, "mech": L.GAME_MECHANICS,
    "app_benefit": L.APP_BENEFITS, "app_feature": L.APP_FEATURES,
    "bot_benefit": L.BOT_BENEFITS, "bot_feature": L.BOT_FEATURES,
    "web_does": L.WEB_DOES, "web_feature": L.WEB_FEATURES,
    "ai_verb": L.AI_VERBS, "ai_feature": L.AI_FEATURES,
    "board": L.HW_BOARDS, "hw_verb": L.HW_VERBS, "hw_feature": L.HW_FEATURES,
    "form": L.CREATIVE_FORMS, "twist": L.CREATIVE_TWISTS,
}
# слоты, которые берутся из словаря темы (ключ в L.THEMES[тема])
THEME_SLOTS = {
    "hero": "heroes", "hero2": "heroes", "place": "places", "goal": "goals", "threat": "threats",
    "about": "about", "fans": "fans",
}
_SLOT_RE = re.compile(r"\{(\w+)\}")


def make_sample(rng: random.Random, kind: Optional[str] = None, themes=None) -> Sample:
    kind = kind or rng.choice(L.KINDS)
    if themes is None:
        themes = [rng.choice(THEME_NAMES)]
        if rng.random() < MASHUP_PROB:
            themes.append(rng.choice([t for t in THEME_NAMES if t != themes[0]]))
    themes = tuple(themes)
    picker = _Picker(rng, themes)
    template = rng.choice(TEMPLATES[kind])
    values = {"title": make_title(rng, picker)}
    for slot in _SLOT_RE.findall(template):
        if slot == "title":
            continue
        if slot == "hero2":
            values[slot] = picker.get_other("heroes", values["hero"])
        elif slot in THEME_SLOTS:
            values[slot] = picker.get(THEME_SLOTS[slot])
        else:
            values[slot] = rng.choice(GLOBAL_SLOTS[slot])
    return Sample(kind, themes, template.format(**values).lower())


# --- Проверка готовой идеи ------------------------------------------------------
def _alternation(items):
    return "|".join(re.escape(item.lower()) for item in sorted(set(items), key=len, reverse=True))


def _slot_items(slot):
    if slot in GLOBAL_SLOTS:
        return GLOBAL_SLOTS[slot]
    return [item for data in L.THEMES.values() for item in data[THEME_SLOTS[slot]]]


def template_regex(template):
    """Регулярное выражение, которому подходит только текст, собранный из фраз словаря по этому шаблону."""
    pattern = []
    for part in re.split(r"(\{\w+\})", template):
        slot = part[1:-1] if _SLOT_RE.fullmatch(part) else None
        if slot == "title":
            pattern.append(r"(?P<title>[^«»]{2,60})")
        elif slot:
            pattern.append(f"(?P<{slot}>{_alternation(_slot_items(slot))})")
        else:
            pattern.append(re.escape(part.lower()))
    return re.compile("".join(pattern))


def valid_titles():
    """Все грамматически правильные названия: «прил. + сущ.», «сущ. + сущ. в род. п.», «сущ. и сущ.», «сущ.»."""
    nouns = {(noun, gender) for data in L.THEMES.values() for noun, gender in data["nouns"]}
    words = {noun for noun, _ in nouns}
    tails = {tail for data in L.THEMES.values() for tail in data["tails"]}
    titles = set(words)
    titles.update(f"{agree(stem, adj_type, gender)} {noun}"
                  for adj_type, stems in L.ADJECTIVES.items() for stem in stems for noun, gender in nouns)
    titles.update(f"{noun} {tail}" for noun in words for tail in tails)
    titles.update(f"{first} и {second}" for first in words for second in words if first != second)
    return titles


_compiled = {}


def _checkers():
    if not _compiled:
        _compiled["templates"] = {kind: [template_regex(t) for t in templates] for kind, templates in TEMPLATES.items()}
        _compiled["titles"] = valid_titles()
    return _compiled


def parse_idea(kind, text):
    """Возвращает (совпадение, причина_отказа). Совпадение есть, только если текст — идея нужного вида,
    целиком собранная из фраз словаря с правильным названием."""
    checkers = _checkers()
    for regex in checkers["templates"][kind]:
        match = regex.fullmatch(text)
        if match:
            break
    else:
        return None, "нарушена структура или фраза не из словаря"
    if match.group("title") not in checkers["titles"]:
        return None, "странное название"
    groups = match.groupdict()
    if "hero2" in groups and groups["hero"] == groups["hero2"]:
        return None, "одинаковые герои"
    return match, ""


def make_corpus(n, seed):
    rng = random.Random(seed)
    return [make_sample(rng) for _ in range(n)]


def lexicon_texts():
    """Все авторские строки словаря — нужны, чтобы словарь токенов покрывал всё."""
    for name in dir(L):
        value = getattr(L, name)
        if name.isupper() and isinstance(value, list) and value and isinstance(value[0], str):
            yield from value
    for theme in L.THEMES.values():
        for key, items in theme.items():
            if key == "keys":
                continue
            for item in items:
                yield item[0] if isinstance(item, tuple) else item
    yield from all_adjective_forms()


def vocabulary_texts():
    """Тексты для построения словаря токенов: словарь + выборка сгенерированных идей (шаблонные слова)."""
    yield from lexicon_texts()
    for sample in make_corpus(3000, seed=12345):
        yield sample.text
