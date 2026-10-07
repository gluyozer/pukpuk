"""Идея: структура и красивый вывод."""
import re
import textwrap
from dataclasses import dataclass
from typing import Tuple

from .lexicon import KIND_LABELS
from .tokenizer import detokenize

BULLET_PREFIXES = ("Для кого:", "Фишка:")
_TITLE_RE = re.compile(r"^«(.+?)» — (.*)$", re.S)


@dataclass(frozen=True)
class Idea:
    kind: str
    themes: Tuple[str, ...]
    title: str
    sentences: Tuple[str, ...]

    @classmethod
    def from_tokens(cls, kind, themes, tokens):
        match = _TITLE_RE.match(detokenize(tokens))
        if not match:
            raise ValueError("в тексте нет названия")
        title, body = match.groups()
        return cls(kind, tuple(themes), title, tuple(re.split(r"(?<=\.)\s+", body.strip())))

    def format(self, width=78):
        emoji, label = KIND_LABELS[self.kind]
        header = f"{emoji} {label} · «{self.title}»"
        lines = [f"{header}   [{' + '.join(self.themes)}]"]
        prose = [s for s in self.sentences if not s.startswith(BULLET_PREFIXES)]
        bullets = [s for s in self.sentences if s.startswith(BULLET_PREFIXES)]
        if prose:
            lines.append(textwrap.fill(" ".join(prose), width, initial_indent="   ", subsequent_indent="   "))
        for bullet in bullets:
            lines.append(textwrap.fill(bullet, width, initial_indent="   • ", subsequent_indent="     "))
        return "\n".join(lines)

    def one_line(self):
        return f"«{self.title}» — " + " ".join(self.sentences)
