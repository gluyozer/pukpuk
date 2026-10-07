import re
import unittest

from ideabot import corpus
from ideabot import lexicon as L
from ideabot.generate import check_idea
from ideabot.tokenizer import PROPER, Vocab, detokenize, tokenize

THEME_LIST_KEYS = ("nouns", "tails", "heroes", "places", "goals", "threats", "about", "fans")


class LexiconTest(unittest.TestCase):
    def test_every_theme_is_complete(self):
        self.assertGreaterEqual(len(L.THEMES), 15)
        for name, data in L.THEMES.items():
            for key in THEME_LIST_KEYS + ("keys",):
                self.assertIn(key, data, f"{name}: нет списка {key}")
                self.assertGreaterEqual(len(data[key]), 3, f"{name}.{key} слишком короткий")
                items = [i[0] if isinstance(i, tuple) else i for i in data[key]]
                self.assertEqual(len(items), len(set(items)), f"{name}.{key}: дубликаты")
                self.assertTrue(all(items) and all(i == i.strip() for i in items), f"{name}.{key}: пустые/с пробелами")
            for word, gender in data["nouns"]:
                self.assertIn(gender, "mfn", f"{name}: род у «{word}»")
            for key in data["keys"]:
                re.compile(key)

    def test_general_lists_have_no_duplicates(self):
        for attr in dir(L):
            value = getattr(L, attr)
            if attr.isupper() and isinstance(value, list) and value and isinstance(value[0], str):
                self.assertEqual(len(value), len(set(value)), attr)

    def test_adjective_agreement(self):
        cases = [("стар", "hard", "f", "старая"), ("стар", "hard", "n", "старое"), ("тих", "velar", "m", "тихий"),
                 ("последн", "soft", "n", "последнее"), ("последн", "soft", "f", "последняя"),
                 ("ночн", "oy", "m", "ночной"), ("ночн", "oy", "f", "ночная"), ("свеж", "hush", "n", "свежее"),
                 ("больш", "oy", "n", "большое")]
        for stem, kind, gender, expected in cases:
            self.assertEqual(corpus.agree(stem, kind, gender), expected)


class CorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = corpus.make_corpus(3000, seed=2024)

    def test_all_kinds_and_mashups_are_generated(self):
        self.assertEqual({s.kind for s in self.samples}, set(L.KINDS))
        self.assertTrue(any(len(s.themes) == 2 for s in self.samples))
        self.assertTrue(any(len(s.themes) == 1 for s in self.samples))

    def test_corpus_passes_the_generators_own_validator(self):
        """Шаблоны корпуса и проверка структуры согласованы: чистые данные проходят валидацию."""
        for s in self.samples:
            ok, why = check_idea(s.kind, tokenize(s.text))
            self.assertTrue(ok, f"{why}: {s.text}")

    def test_every_template_slot_is_defined(self):
        slot_re = re.compile(r"\{(\w+)\}")
        for kind, templates in corpus.TEMPLATES.items():
            for template in templates:
                for slot in slot_re.findall(template):
                    self.assertTrue(slot == "title" or slot in corpus.GLOBAL_SLOTS or slot in corpus.THEME_SLOTS,
                                    f"{kind}: неизвестный слот {slot}")

    def test_titles_from_the_corpus_are_grammatical(self):
        titles = corpus.valid_titles()
        self.assertGreater(len(titles), 20000)
        for s in self.samples:
            title = re.match(r"«([^»]+)»", s.text).group(1)
            self.assertIn(title, titles)

    def test_validator_rejects_grammar_blends(self):
        good = ("«тихий маяк» — рогалик. ты — юный капитан в тайной бухте, и тебе нужно найти дорогу домой, "
                "пока надвигается шторм. фишка: карта каждый раз генерируется заново.")
        self.assertTrue(check_idea("игра", tokenize(good))[0])
        blends = {
            # реальная ошибка нейросети: «в городе, которое…» (склейка двух фраз)
            "склейка фраз": good.replace("в тайной бухте", "в городе, которое забыла карта"),
            "дубль в названии": good.replace("тихий маяк", "спутник и спутник"),
            "неверное согласование названия": good.replace("тихий маяк", "тихая маяк"),
            "выдуманный жанр": good.replace("рогалик", "нечто"),
            "обрезано": good[:-30],
            "нет названия": good.split("» — ", 1)[1],
            "два блока фишки": good + " фишка: ещё одна.",
            "не тот вид": good.replace("— рогалик", "— telegram-бот"),
        }
        for name, text in blends.items():
            self.assertFalse(check_idea("игра", tokenize(text))[0], name)
        self.assertFalse(check_idea("бот", tokenize(good))[0])

    def test_two_heroes_in_coop_template_differ(self):
        text = ("«тихий маяк» — кооперативная головоломка на краю галактики: один игрок — юный астроном, "
                "другой — юный астроном. вместе им нужно найти дорогу домой, пока кончается кислород. "
                "фишка: карта каждый раз генерируется заново.")
        self.assertEqual(check_idea("игра", tokenize(text)), (False, "одинаковые герои"))
        self.assertTrue(check_idea("игра", tokenize(text.replace("другой — юный астроном", "другой — робот-механик с разбитой станции")))[0])

    def test_no_unknown_tokens(self):
        vocab = Vocab.build(corpus.vocabulary_texts())
        for s in self.samples:
            self.assertNotIn(vocab.unk_id, vocab.encode(s.tokens()), s.text)

    def test_deterministic_for_a_seed(self):
        self.assertEqual(corpus.make_corpus(50, seed=5), corpus.make_corpus(50, seed=5))
        self.assertNotEqual(corpus.make_corpus(50, seed=5), corpus.make_corpus(50, seed=6))

    def test_forced_kind_and_themes(self):
        import random
        sample = corpus.make_sample(random.Random(1), kind="бот", themes=["море", "роботы"])
        self.assertEqual((sample.kind, sample.themes), ("бот", ("море", "роботы")))
        self.assertIn("telegram-бот", sample.text)


class TokenizerTest(unittest.TestCase):
    def test_every_lexicon_phrase_survives_tokenization(self):
        """Фраза словаря после токенизации и сборки не меняется — иначе проверка идей не узнает её."""
        phrases = list(corpus.lexicon_texts())
        self.assertGreater(len(phrases), 1000)
        for phrase in phrases:
            self.assertEqual(detokenize(tokenize(phrase), pretty=False), phrase.lower(), phrase)

    def test_roundtrip_is_exact(self):
        # токенизатор приводит текст к нижнему регистру, больше ничего не теряется
        for s in corpus.make_corpus(500, seed=3):
            self.assertEqual(detokenize(tokenize(s.text), pretty=False), s.text.lower())

    def test_proper_names_are_restored(self):
        """Каждое слово с заглавными буквами в словаре (Марс, Arduino, ESP32, Wi-Fi ...) вернётся при форматировании."""
        found = set()
        for phrase in corpus.lexicon_texts():
            for word in re.findall(r"\w+(?:-\w+)*", phrase):
                if any(c.isupper() for c in word):
                    found.add(word)
                    self.assertEqual(PROPER.get(word.lower()), word, f"нет записи в PROPER для «{word}»")
        self.assertTrue({"Марс", "Arduino", "ESP32", "Wi-Fi"} <= found)  # тест действительно что-то проверяет

    def test_sentence_after_title_is_capitalized(self):
        text = detokenize(tokenize("«тихий маяк» — творческий проект: комикс про Марс. фишка: всё за один день."))
        self.assertEqual(text, "«Тихий Маяк» — Творческий проект: комикс про Марс. Фишка: всё за один день.")

    def test_vocab_serialization(self):
        vocab = Vocab.build(["привет мир", "мир тесен"])
        again = Vocab.from_json(vocab.to_json())
        self.assertEqual(vocab.tokens, again.tokens)
        self.assertEqual(vocab.encode(["мир", "нет-такого"]), again.encode(["мир", "нет-такого"]))
        self.assertEqual(vocab.encode(["нет-такого"]), [vocab.unk_id])


if __name__ == "__main__":
    unittest.main()
