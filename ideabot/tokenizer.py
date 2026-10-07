"""Токенизация текста идей и словарь модели."""
import json
import re

from .lexicon import KINDS, THEMES

PAD, UNK, END, NONE = "<pad>", "<unk>", "<end>", "<нет>"
SPECIAL_TOKENS = [PAD, UNK, END, NONE]

TOKEN_RE = re.compile(r"[«»—:,.!?;]|[a-zа-яё0-9]+(?:-[a-zа-яё0-9]+)*")
NO_SPACE_BEFORE = {",", ".", ":", ";", "!", "?", "»"}
NO_SPACE_AFTER = {"«"}
SENTENCE_END = {".", "!", "?"}
SMALL_WORDS = {"и", "в", "на", "из", "для", "без", "о", "с", "к", "у", "от", "до", "за"}
PROPER = {
    "telegram-бот": "Telegram-бот", "telegram": "Telegram", "arduino": "Arduino", "raspberry": "Raspberry",
    "pi": "Pi", "esp32": "ESP32", "diy": "DIY", "ии": "ИИ", "ии-проект": "ИИ-проект", "wi-fi": "Wi-Fi",
    "qr-код": "QR-код", "3d-принтере": "3D-принтере", "марс": "Марс", "луны": "Луны",
}


def kind_token(kind):
    return f"<{kind}>"


def theme_token(theme):
    return f"<т:{theme}>"


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def _cap(word):
    return word[:1].upper() + word[1:]


def detokenize(tokens, pretty=True):
    """Собирает текст из токенов. pretty=True — заглавные буквы и «Названия» с большой буквы."""
    out = []
    start = True
    in_title = False
    first_in_title = False
    prev = None
    for tok in tokens:
        if tok == "«":
            in_title, first_in_title = True, True
        elif tok == "»":
            in_title = False
        word = tok
        if pretty and tok[:1].isalnum():
            if tok in PROPER:
                word = PROPER[tok]
            elif start or (in_title and (first_in_title or tok not in SMALL_WORDS)):
                word = _cap(tok)
            first_in_title = False
            start = False
        elif pretty and (tok in SENTENCE_END or (tok == "—" and prev == "»")):
            start = True  # после точки и после «название» — с большой буквы
        if prev is not None and tok not in NO_SPACE_BEFORE and prev not in NO_SPACE_AFTER:
            out.append(" ")
        out.append(word)
        prev = tok
    return "".join(out)


class Vocab:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.pad_id = self.index[PAD]
        self.unk_id = self.index[UNK]
        self.end_id = self.index[END]
        self.forbidden_ids = [self.index[t] for t in self.tokens if _is_control(t)]

    def __len__(self):
        return len(self.tokens)

    def encode(self, tokens):
        return [self.index.get(t, self.unk_id) for t in tokens]

    def decode(self, ids):
        return [self.tokens[i] for i in ids]

    def to_json(self):
        return json.dumps(self.tokens, ensure_ascii=False)

    @classmethod
    def from_json(cls, text):
        return cls(json.loads(text))

    @classmethod
    def build(cls, texts):
        control = SPECIAL_TOKENS + [kind_token(k) for k in KINDS] + [theme_token(t) for t in THEMES]
        words = set()
        for text in texts:
            words.update(tokenize(text))
        return cls(control + sorted(words - set(control)))


def _is_control(token):
    """Служебные токены (<pad>, <игра>, <т:космос> ...) не должны попадать в текст идеи. <end> — можно."""
    return token.startswith("<") and token.endswith(">") and token != END
