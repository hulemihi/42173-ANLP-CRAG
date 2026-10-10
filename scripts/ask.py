"""
Ask the RAG pipeline medical questions, as a user would.

    python scripts/ask.py                                   # interactive: type questions, empty line to quit
    python scripts/ask.py -q "Does vitamin C cure colds?"   # one or more questions (repeat -q)
    python scripts/ask.py --model 7b                        # Qwen2.5-7B in 4-bit
    python scripts/ask.py --show-retrieved                  # also print the retriever's scores
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crag.data import load_textbooks_corpus  # noqa: E402
from crag.generator import answer_question, load_generator  # noqa: E402
from crag.retriever import build_index, retrieve_with_scores  # noqa: E402

MODELS = {
    "0.5b": dict(model_name="Qwen/Qwen2.5-0.5B-Instruct"),
    "3b": dict(model_name="Qwen/Qwen2.5-3B-Instruct"),
    "7b": dict(model_name="Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True),
}
INDEX_PATH = ROOT / "indexes" / "contriever-msmarco_textbooks.pt"


def answer(question, index, model, tokenizer, k, show_retrieved):
    hits = retrieve_with_scores(index, question, k=k)
    if show_retrieved:
        print("\nRetrieved:")
        for p, s in hits:
            print(f"  [{s:.2f}] {p[:120]}...")
    out = answer_question(model, tokenizer, question, [p for p, _ in hits])
    print("\n" + out.response + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", action="append", help="question to ask (repeat for several)")
    parser.add_argument("--model", choices=MODELS, default="3b")
    parser.add_argument("--k", type=int, default=5, help="passages retrieved per question")
    parser.add_argument("--show-retrieved", action="store_true")
    args = parser.parse_args()

    print("Loading retriever index and generator...")
    index = build_index(load_textbooks_corpus(), cache_path=INDEX_PATH)
    model, tokenizer = load_generator(**MODELS[args.model])

    if args.question:
        for q in args.question:
            print("#" * 70 + f"\nQ: {q}")
            answer(q, index, model, tokenizer, args.k, args.show_retrieved)
        return

    print("Ask a medical question (empty line to quit).")
    while True:
        try:
            q = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        answer(q, index, model, tokenizer, args.k, args.show_retrieved)


if __name__ == "__main__":
    main()
