"""Консольный чат-бот: принимает команды и обычные фразы, отвечает идеями от нейросети."""
import re
from pathlib import Path

from .generate import DEFAULT_TEMPERATURE, IdeaGenerator
from .lexicon import KIND_LABELS
from .request import Request, parse_request

HELP = """Я — ИдеяБот, нейросеть, которая придумывает идеи для игр и разных проектов.

Просто напиши, что нужно, например:
  придумай игру про космос и котов
  3 идеи для приложения про еду
  хочу бота про музыку
  идею для проекта

Команды:
  /игра [тема]        идея игры              /сайт [тема]       идея веб-сервиса
  /приложение [тема]  идея приложения        /ии [тема]         идея ИИ-проекта
  /бот [тема]         идея Telegram-бота     /железо [тема]     идея DIY-устройства
  /творчество [тема]  идея творческого проекта
  /идея [тема]        любой вид проекта      /ещё               повторить последний запрос
  /темы               список тем             /сохранить         записать идеи в favorites.md
  /креатив [0.3-1.5]  смелость идей (чем выше, тем страннее)
  /помощь             эта подсказка          /выход             закончить

Можно задать до двух тем сразу — получится смесь, например «игра про море и роботов»."""

COMMAND_KINDS = {
    "игра": "игра", "game": "игра", "приложение": "приложение", "апп": "приложение", "app": "приложение",
    "бот": "бот", "bot": "бот", "сайт": "сайт", "веб": "сайт", "ии": "ии", "ai": "ии", "нейросеть": "ии",
    "железо": "железо", "diy": "железо", "творчество": "творчество",
}
EXIT_COMMANDS = {"выход", "exit", "quit", "q", "пока"}
HELP_COMMANDS = {"помощь", "help", "start", "?", "h"}
AGAIN_COMMANDS = {"ещё", "еще", "more", "again", "повтори"}
SAVE_COMMANDS = {"сохранить", "save", "fav", "избранное"}
TEMP_COMMANDS = {"креатив", "температура", "temp"}
THEMES_COMMANDS = {"темы", "themes"}

GREETING_RE = re.compile(r"^\s*(привет\w*|здравствуй\w*|добр\w+ (?:день|вечер|утро)|хай|салют|hello|hi)\b", re.I)
THANKS_RE = re.compile(r"^\s*(спасибо|благодарю|thanks|класс|круто|супер)\b", re.I)
HELP_RE = re.compile(r"(что ты умеешь|как тобой пользоваться|помоги мне|^помощь$)", re.I)


class Chat:
    def __init__(self, generator: IdeaGenerator, read=input, write=print, favorites_path="favorites.md",
                 temperature=DEFAULT_TEMPERATURE):
        self.generator = generator
        self.read, self.write = read, write
        self.favorites_path = Path(favorites_path)
        self.temperature = temperature
        self.last_request = None
        self.last_ideas = []
        self.shown_titles = []

    # ---------------------------------------------------------------- цикл
    def run(self):
        from .corpus import THEME_NAMES
        self.say("Привет! Я ИдеяБот 🤖 — придумываю идеи для игр и проектов. Напиши, что нужно, "
                 "или /помощь.\nТемы: " + ", ".join(THEME_NAMES))
        while True:
            try:
                line = self.read("\nВы: ")
            except (EOFError, KeyboardInterrupt):
                self.say("\nДо встречи!")
                return
            if not self.handle(line):
                return

    def say(self, text):
        self.write(text)

    # ---------------------------------------------------------------- обработка сообщения
    def handle(self, line):
        """Обрабатывает одно сообщение. Возвращает False, если пора выходить."""
        line = line.strip()
        if not line:
            return True
        if line.startswith("/"):
            return self._command(line[1:])
        if GREETING_RE.match(line) and not parse_request(line).themes:
            self.say("Привет! Скажи, что придумать: игру, приложение, бота… и тему — например «игра про космос».")
        elif THANKS_RE.match(line):
            self.say("Всегда пожалуйста! Хочешь ещё идею? Напиши /ещё.")
        elif HELP_RE.search(line):
            self.say(HELP)
        else:
            self._idea(parse_request(line))
        return True

    def _command(self, text):
        name, _, rest = text.partition(" ")
        name = name.lower().split("@")[0]
        rest = rest.strip()
        if name in EXIT_COMMANDS:
            self.say("До встречи! Удачных проектов 🚀")
            return False
        if name in HELP_COMMANDS:
            self.say(HELP)
        elif name in THEMES_COMMANDS:
            from .corpus import THEME_NAMES
            self.say("Темы: " + ", ".join(THEME_NAMES) + "\nМожно взять одну или две сразу.")
        elif name in AGAIN_COMMANDS:
            if self.last_request is None:
                self.say("Пока нечего повторять — сначала попроси идею.")
            else:
                self._idea(self.last_request)
        elif name in SAVE_COMMANDS:
            self._save()
        elif name in TEMP_COMMANDS:
            self._set_temperature(rest)
        elif name == "идея":
            self._idea(parse_request(rest))
        elif name in COMMAND_KINDS:
            self._idea(parse_request(rest, kind=COMMAND_KINDS[name]))
        else:
            self.say(f"Не знаю команду /{name}. Список команд — /помощь.")
        return True

    # ---------------------------------------------------------------- действия
    def _idea(self, request: Request):
        self.last_request = request
        if request.unknown and not request.themes:
            words = ", ".join(f"«{w}»" for w in request.unknown[:3])
            self.say(f"Про {words} я пока ничего не знаю, поэтому придумаю что-нибудь на другую тему "
                     f"(список тем — /темы).")
        ideas = self.generator.generate(n=request.count, kind=request.kind, themes=request.themes or None,
                                        temperature=self.temperature, exclude=self.shown_titles)
        if not ideas:
            self.say("Не получилось собрать удачную идею — попробуй ещё раз или измени /креатив.")
            return
        self.last_ideas = ideas
        self.shown_titles.extend(i.title for i in ideas)
        self.say("")
        self.say("\n\n".join(i.format() for i in ideas))

    def _save(self):
        if not self.last_ideas:
            self.say("Пока нечего сохранять.")
            return
        with self.favorites_path.open("a", encoding="utf-8") as f:
            for idea in self.last_ideas:
                f.write(f"## {KIND_LABELS[idea.kind][1]}: «{idea.title}»\n\n")
                f.write(" ".join(idea.sentences) + "\n\n")
        self.say(f"Сохранил {len(self.last_ideas)} шт. в {self.favorites_path}")

    def _set_temperature(self, text):
        if not text:
            self.say(f"Сейчас креативность {self.temperature:.2f}. Задать: /креатив 0.3 … 1.5")
            return
        try:
            value = float(text.replace(",", "."))
        except ValueError:
            self.say("Нужно число, например /креатив 1.1")
            return
        self.temperature = min(1.5, max(0.3, value))
        self.say(f"Креативность: {self.temperature:.2f} (выше — смелее и страннее, ниже — аккуратнее).")
