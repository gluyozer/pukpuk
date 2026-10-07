import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args, stdin=None, cwd=None):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-m", "ideabot", *args], input=stdin, capture_output=True, text=True,
                          encoding="utf-8", env=env, cwd=cwd or tempfile.mkdtemp(), timeout=120)


class CliTest(unittest.TestCase):
    def test_idea_command(self):
        result = run_cli("idea", "-k", "бот", "-t", "еда", "-n", "2", "--seed", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("🤖 Telegram-бот · «"), 2)
        self.assertIn("[еда]", result.stdout)

    def test_idea_is_reproducible_with_seed(self):
        a = run_cli("idea", "-n", "2", "--seed", "5").stdout
        b = run_cli("idea", "-n", "2", "--seed", "5").stdout
        self.assertEqual(a, b)
        self.assertNotEqual(a, run_cli("idea", "-n", "2", "--seed", "6").stdout)

    def test_free_form_topic_with_mashup(self):
        result = run_cli("idea", "-k", "игра", "-t", "космос и котов", "--seed", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[космос + животные]", result.stdout)

    def test_unknown_topic_warns_on_stderr(self):
        result = run_cli("idea", "-t", "оригами", "--seed", "1")
        self.assertEqual(result.returncode, 0)
        self.assertIn("оригами", result.stderr)

    def test_chat_over_a_pipe(self):
        result = run_cli("chat", "--seed", "3", stdin="привет\n/игра космос\n/выход\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Привет! Я ИдеяБот", result.stdout)
        self.assertIn("🎮 Игра · «", result.stdout)
        self.assertIn("До встречи", result.stdout)

    def test_default_command_is_chat_and_eof_is_fine(self):
        result = run_cli(stdin="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ИдеяБот", result.stdout)

    def test_bad_arguments(self):
        self.assertNotEqual(run_cli("idea", "-k", "несуществующий").returncode, 0)


if __name__ == "__main__":
    unittest.main()
