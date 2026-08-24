#!/usr/bin/env python3
"""Stage 4 — try the trained model. Single shot or interactive REPL.

  python scripts/04_infer.py --ckpt artifacts/thinkspark/best \
      --input "yaar ye refund abhi tak nahi aaya, kitni baar bolu?" \
      --context "user is angrily chasing a delayed refund"

  python scripts/04_infer.py --ckpt artifacts/thinkspark/best   # interactive
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import FILLER_DICT_JSON  # noqa: E402
from thinkspark.infer import ThinkSparkPredictor  # noqa: E402


def show(res: dict) -> None:
    spark = res["spark"] if res["spark"] else "· (silence)"
    print(f"\n  spark    : {spark}")
    print(f"  intent   : {res['intent']}  (p={res['confidence']['intent']:.2f})")
    print(f"  language : {res['language']}  (p={res['confidence']['language']:.2f})")
    print(f"  register : {res['register']} | emotion: {res['emotion']} | "
          f"type: {res['filler_type']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/thinkspark/best")
    ap.add_argument("--input", default="")
    ap.add_argument("--context", default="")
    ap.add_argument("--lang", default=None, help="force output language code")
    args = ap.parse_args()

    pred = ThinkSparkPredictor(args.ckpt, FILLER_DICT_JSON)

    if args.input:
        show(pred.predict(args.input, args.context, force_lang=args.lang))
        return 0

    print("ThinkSpark REPL — type the user's line. Prefix context with '@'. Ctrl-D to quit.")
    context = args.context
    try:
        while True:
            line = input("\nuser> ").strip()
            if not line:
                continue
            if line.startswith("@"):
                context = line[1:].strip()
                print(f"  (context set: {context!r})")
                continue
            show(pred.predict(line, context, force_lang=args.lang))
    except (EOFError, KeyboardInterrupt):
        print("\nbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
