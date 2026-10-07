import unittest

from ideabot import corpus
from ideabot import lexicon as L
from ideabot.evaluate import theme_words
from ideabot.generate import IdeaGenerator, check_idea
from ideabot.request import normalize_words
from ideabot.tokenizer import Vocab, tokenize
from ideabot.train import evaluate as val_loss


class TrainedModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = IdeaGenerator.load(seed=1)

    def test_committed_model_matches_current_vocabulary(self):
        """Если изменили словарь (lexicon.py/corpus.py), но не переобучили модель — тест это поймает."""
        current = Vocab.build(corpus.vocabulary_texts())
        self.assertEqual(self.generator.vocab.tokens, current.tokens,
                         "словарь изменился — переобучите модель: python -m ideabot train")

    def test_model_quality_on_fresh_corpus(self):
        held_out = corpus.make_corpus(400, seed=31337)
        loss = val_loss(self.generator.model, self.generator.vocab, held_out)
        self.assertLess(loss, 0.65, "модель обучена плохо")

    def test_every_kind_gives_valid_ideas_on_a_topic(self):
        for kind in L.KINDS:
            ideas = self.generator.generate(n=3, kind=kind, themes=["космос"])
            self.assertEqual(len(ideas), 3, kind)
            for idea in ideas:
                self.assertEqual(idea.kind, kind)
                ok, why = check_idea(kind, tokenize(idea.one_line()))
                self.assertTrue(ok, f"{kind}: {why}: {idea.one_line()}")

    def test_topic_is_respected(self):
        for theme in ("космос", "еда", "динозавры", "музыка"):
            words = theme_words(theme)
            ideas = self.generator.generate(n=6, themes=[theme])
            self.assertEqual(len(ideas), 6)
            hits = sum(bool(words & {w[:5] for w in normalize_words(i.one_line())}) for i in ideas)
            self.assertGreaterEqual(hits, 5, f"{theme}: тема попала только в {hits} из 6")

    def test_mashup_of_two_themes(self):
        ideas = self.generator.generate(n=4, kind="игра", themes=["море", "роботы"])
        self.assertEqual(len(ideas), 4)
        self.assertTrue(all(i.themes == ("море", "роботы") for i in ideas))

    def test_random_mode_is_varied(self):
        ideas = self.generator.generate(n=20)
        self.assertEqual(len(ideas), 20)
        self.assertGreaterEqual(len({i.kind for i in ideas}), 4)
        self.assertGreaterEqual(len({i.themes for i in ideas}), 8)
        self.assertEqual(len({i.title for i in ideas}), 20, "названия не должны повторяться")

    def test_same_seed_same_ideas(self):
        a = IdeaGenerator.load(seed=42).generate(n=3, kind="игра")
        b = IdeaGenerator.load(seed=42).generate(n=3, kind="игра")
        c = IdeaGenerator.load(seed=43).generate(n=3, kind="игра")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_exclude_titles(self):
        first = self.generator.generate(n=5, kind="бот", themes=["еда"])
        again = self.generator.generate(n=5, kind="бот", themes=["еда"], exclude=[i.title for i in first])
        self.assertFalse({i.title.lower() for i in first} & {i.title.lower() for i in again})

    def test_extreme_temperatures_do_not_crash(self):
        for temperature in (0.0, 0.3, 1.5):
            ideas = self.generator.generate(n=2, kind="приложение", temperature=temperature)
            self.assertLessEqual(len(ideas), 2)
        self.assertEqual(len(self.generator.generate(n=2, kind="приложение", temperature=0.3)), 2)

    def test_no_grammar_blends_in_output(self):
        """Всё, что выдаёт generate(), состоит из фраз словаря (нет склеек вроде «в городе, которое…»)."""
        for idea in self.generator.generate(n=60, max_rounds=20):
            self.assertTrue(check_idea(idea.kind, tokenize(idea.one_line()))[0], idea.one_line())

    def test_ideas_are_not_just_copies_of_training_data(self):
        reference = {s.text for s in corpus.make_corpus(30000, seed=777)}
        ideas = self.generator.generate(n=30)
        copies = sum(i.one_line().lower() in reference for i in ideas)
        self.assertLess(copies, 5)

    def test_formatting(self):
        idea = self.generator.generate(n=1, kind="игра", themes=["космос"])[0]
        text = idea.format()
        self.assertTrue(text.startswith("🎮 Игра · «"))
        self.assertIn("[космос]", text)
        self.assertIn("• Фишка:", text)
        self.assertNotIn("<", text)
        self.assertTrue(all(len(line) <= 90 for line in text.splitlines()))


if __name__ == "__main__":
    unittest.main()
