"""Точка входа:  python -m ideabot [chat | idea | train | eval]"""
import argparse
import sys

from . import __version__
from .corpus import THEME_NAMES
from .lexicon import KINDS
from .request import parse_request


def _load_generator(seed=None):
    from .generate import IdeaGenerator
    from .train import MODEL_PATH
    if not MODEL_PATH.exists():
        print("Модель ещё не обучена — обучаю (около 5 минут, это делается один раз)...", file=sys.stderr)
        from .train import train
        train(log=lambda m: print(m, file=sys.stderr))
    return IdeaGenerator.load(seed=seed)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ideabot", description="Нейросеть-бот, которая придумывает идеи для игр и проектов.")
    parser.add_argument("--version", action="version", version=f"ideabot {__version__}")
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="интерактивный чат (по умолчанию)")
    chat.add_argument("--seed", type=int, default=None)
    chat.add_argument("--favorites", default="favorites.md", help="куда сохранять командой /сохранить")

    idea = sub.add_parser("idea", help="придумать идеи и выйти")
    idea.add_argument("-k", "--kind", choices=KINDS, help="вид проекта (по умолчанию случайный)")
    idea.add_argument("-t", "--topic", default="", help=f"тема своими словами или из списка: {', '.join(THEME_NAMES)}")
    idea.add_argument("-n", "--count", type=int, default=1)
    idea.add_argument("--seed", type=int, default=None)
    idea.add_argument("--temperature", type=float, default=0.85, help="креативность 0.3–1.5")

    from . import train as training
    train = sub.add_parser("train", help="обучить нейросеть заново")
    train.add_argument("--epochs", type=int, default=training.DEFAULT_EPOCHS)
    train.add_argument("--seqs", type=int, default=training.DEFAULT_SEQS_PER_EPOCH, help="идей в эпоху")
    train.add_argument("--hidden", type=int, default=training.DEFAULT_HIDDEN)
    train.add_argument("--seed", type=int, default=0)

    sub.add_parser("eval", help="проверить качество модели")

    args = parser.parse_args(argv)
    command = args.command or "chat"

    if command == "train":
        from .train import train as run_train
        run_train(epochs=args.epochs, seqs_per_epoch=args.seqs, hidden=args.hidden, seed=args.seed)
        return 0
    if command == "eval":
        from .evaluate import evaluate
        evaluate(_load_generator(seed=0))
        return 0
    if command == "idea":
        generator = _load_generator(args.seed)
        request = parse_request(args.topic, kind=args.kind)
        if request.unknown and not request.themes:
            print(f"(тема «{' '.join(request.unknown)}» неизвестна — беру случайную; темы: {', '.join(THEME_NAMES)})",
                  file=sys.stderr)
        ideas = generator.generate(n=max(1, args.count), kind=request.kind, themes=request.themes or None,
                                   temperature=min(1.5, max(0.3, args.temperature)))
        if not ideas:
            print("Не удалось собрать идею, попробуйте ещё раз.", file=sys.stderr)
            return 1
        print("\n\n".join(i.format() for i in ideas))
        return 0

    from .bot import Chat
    seed = getattr(args, "seed", None)
    Chat(_load_generator(seed), favorites_path=getattr(args, "favorites", "favorites.md")).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
