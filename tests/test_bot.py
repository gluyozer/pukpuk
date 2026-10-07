import tempfile
import unittest
from pathlib import Path

from ideabot.bot import Chat
from ideabot.generate import IdeaGenerator


class Script:
    """Подставляет заготовленные реплики пользователя и собирает ответы бота."""

    def __init__(self, lines):
        self.lines = list(lines)
        self.output = []

    def read(self, prompt=""):
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)

    def write(self, text=""):
        self.output.append(str(text))

    @property
    def text(self):
        return "\n".join(self.output)


class ChatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = IdeaGenerator.load(seed=7)

    def chat(self, lines, favorites=None):
        script = Script(lines)
        favorites = favorites or Path(tempfile.mkdtemp()) / "fav.md"
        bot = Chat(self.generator, read=script.read, write=script.write, favorites_path=favorites)
        bot.run()
        return bot, script

    def test_full_session(self):
        bot, out = self.chat(["привет", "/темы", "придумай игру про космос и котов", "/ещё", "спасибо", "/выход"])
        self.assertIn("Темы: космос", out.text)
        self.assertEqual(out.text.count("🎮 Игра · «"), 2)
        self.assertIn("[космос + животные]", out.text)
        self.assertIn("Всегда пожалуйста", out.text)
        self.assertIn("До встречи", out.text)
        self.assertEqual(len(bot.shown_titles), 2)
        self.assertEqual(len(set(bot.shown_titles)), 2, "бот не должен повторять названия")

    def test_every_kind_command(self):
        commands = ["/игра", "/приложение", "/бот", "/сайт", "/ии", "/железо", "/творчество", "/идея"]
        _, out = self.chat([f"{c} море" for c in commands])
        for marker in ("🎮 Игра", "📱 Приложение", "🤖 Telegram-бот", "🌐 Веб-сервис", "🧠 ИИ-проект",
                       "🔧 Железо", "🎨 Творческий проект"):
            self.assertIn(marker, out.text)
        self.assertNotIn("Не получилось", out.text)

    def test_free_text_count_and_kind(self):
        _, out = self.chat(["3 идеи для приложения про еду"])
        self.assertEqual(out.text.count("📱 Приложение · «"), 3)

    def test_unknown_topic_is_reported_but_idea_is_given(self):
        _, out = self.chat(["придумай игру про оригами"])
        self.assertIn("«оригами»", out.text)
        self.assertIn("🎮 Игра", out.text)

    def test_save_to_favorites(self):
        favorites = Path(tempfile.mkdtemp()) / "fav.md"
        bot, out = self.chat(["/сохранить", "/бот еда", "/сохранить"], favorites=favorites)
        self.assertIn("Пока нечего сохранять", out.text)
        content = favorites.read_text(encoding="utf-8")
        self.assertIn("## Telegram-бот: «" + bot.last_ideas[0].title + "»", content)

    def test_creativity_command(self):
        bot, out = self.chat(["/креатив 1.2", "/креатив 9", "/креатив abc", "/креатив"])
        self.assertIn("1.20", out.text)
        self.assertAlmostEqual(bot.temperature, 1.5)
        self.assertIn("Нужно число", out.text)

    def test_unknown_command_and_empty_input_and_again_without_history(self):
        _, out = self.chat(["", "   ", "/абракадабра", "/ещё"])
        self.assertIn("Не знаю команду /абракадабра", out.text)
        self.assertIn("Пока нечего повторять", out.text)

    def test_help_and_eof(self):
        _, out = self.chat(["/помощь", "что ты умеешь"])
        self.assertEqual(out.text.count("Команды:"), 2)
        self.assertIn("До встречи", out.text)  # конец ввода (EOF) завершает чат без ошибки


if __name__ == "__main__":
    unittest.main()
