"""
Chat with the medical question-answering system, as a user would.

    python scripts/ask.py                                   # choose RAG or CRAG, then chat
    python scripts/ask.py --mode crag                       # skip the menu
    python scripts/ask.py -q "Does vitamin C cure colds?"   # answer one or more questions and exit (repeat -q)
    python scripts/ask.py --model 7b                        # Qwen2.5-7B in 4-bit
    python scripts/ask.py --show-retrieved                  # also print retrieval scores and the CRAG action

While chatting: /mode switches between RAG and CRAG, /sources toggles the retrieval details, /help lists
commands, /quit (or an empty line) exits. Each question is answered on its own; earlier turns are not remembered.

On start-up, CRAG is checked with pipeline.check_crag_ready(). Until every CRAG stage (actions / refinement /
web_search) is implemented, CRAG can't be selected and the reason is shown.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crag.data import load_textbooks_corpus  # noqa: E402
from crag.evaluator import load_evaluator  # noqa: E402
from crag.generator import load_generator  # noqa: E402
from crag.pipeline import ask_crag, ask_rag, check_crag_ready  # noqa: E402
from crag.retriever import build_index, retrieve_with_scores  # noqa: E402

MODELS = {
    "0.5b": dict(model_name="Qwen/Qwen2.5-0.5B-Instruct"),
    "3b": dict(model_name="Qwen/Qwen2.5-3B-Instruct"),
    "7b": dict(model_name="Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True),
}
INDEX_PATH = ROOT / "indexes" / "contriever-msmarco_textbooks.pt"

MODES = {
    "rag": "RAG  - retrieve textbook passages and answer from them",
    "crag": "CRAG - check the retrieved passages first, filter them or search the web, then answer",
}

HELP = """Commands:
  /mode      switch between RAG and CRAG
  /sources   show or hide the retrieved passages, scores and CRAG action
  /help      show this list
  /quit      exit (an empty line also exits)"""


class System:
    """
    Holds the loaded models and the current mode. The evaluator is only loaded once CRAG is first used.
    """

    def __init__(self, model_key, k, show_retrieved, crag_problems):
        print("Loading the retriever index (first run builds it, which takes a while)...")
        self.index = build_index(load_textbooks_corpus(), cache_path=INDEX_PATH)
        print(f"Loading the generator (Qwen2.5-{model_key.upper()})...")
        self.model, self.tokenizer = load_generator(**MODELS[model_key])
        self.evaluator = None
        self.mode = None
        self.k = k
        self.show_retrieved = show_retrieved
        self.crag_problems = crag_problems  # from check_crag_ready(); empty means CRAG can be used

    def set_mode(self, mode):
        if mode == "crag" and self.evaluator is None:
            print("Loading the retrieval evaluator...")
            self.evaluator = load_evaluator()
        self.mode = mode
        print(f"Mode: {mode.upper()}")

    def answer(self, question):
        """
        Answer one question in the current mode and print the response.
        """
        if self.show_retrieved:
            print("\nRetrieved:")
            for p, s in retrieve_with_scores(self.index, question, k=self.k):
                print(f"  [{s:.2f}] {p[:120]}...")

        if self.mode == "crag":
            out, action, scores = ask_crag(question, self.index, *self.evaluator, self.model, self.tokenizer, k=self.k)
            if self.show_retrieved:
                print(f"Evaluator scores: {[round(s, 2) for s in scores]} -> action: {action}")
        else:
            out = ask_rag(question, self.index, self.model, self.tokenizer, k=self.k)
        print("\n" + out.response)


def print_crag_problems(problems):
    print("CRAG is not available yet:")
    for problem in problems:
        print(f"  - {problem}")


def choose_mode(crag_problems):
    """
    Ask the user to pick RAG or CRAG. Returns "rag" or "crag". CRAG is refused while crag_problems is non-empty.
    """
    print("\nWhich system do you want to use?")
    keys = list(MODES)
    for i, key in enumerate(keys, 1):
        unavailable = "  (not available yet)" if key == "crag" and crag_problems else ""
        print(f"  {i}. {MODES[key]}{unavailable}")
    while True:
        try:
            choice = input(f"Enter 1-{len(keys)}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            choice = keys[int(choice) - 1]
        if choice not in keys:
            print("Please enter 1 or 2.")
        elif choice == "crag" and crag_problems:
            print_crag_problems(crag_problems)
            print("Please choose 1 (RAG) for now.")
        else:
            return choice


def chat(system):
    print("\nAsk a medical question. Type /help for commands, or press Enter on an empty line to quit.")
    while True:
        try:
            q = input(f"\n[{system.mode.upper()}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in ("/quit", "/exit"):
            break
        if q.lower() == "/help":
            print(HELP)
        elif q.lower() == "/mode":
            system.set_mode(choose_mode(system.crag_problems))
        elif q.lower() == "/sources":
            system.show_retrieved = not system.show_retrieved
            print(f"Retrieval details {'shown' if system.show_retrieved else 'hidden'}.")
        elif q.startswith("/"):
            print(f"Unknown command {q}. Type /help for commands.")
        else:
            system.answer(q)
    print("Goodbye.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, help="skip the menu and start in this mode")
    parser.add_argument("-q", "--question", action="append", help="answer these questions and exit (repeat for several)")
    parser.add_argument("--model", choices=MODELS, default="3b")
    parser.add_argument("--k", type=int, default=5, help="passages retrieved per question")
    parser.add_argument("--show-retrieved", action="store_true")
    args = parser.parse_args()

    print("Checking the CRAG stages...")
    crag_problems = check_crag_ready()
    if args.mode == "crag" and crag_problems:
        print_crag_problems(crag_problems)
        print("Run with --mode rag, or without --mode to choose from the menu.")
        sys.exit(1)

    mode = args.mode or choose_mode(crag_problems)
    system = System(args.model, args.k, args.show_retrieved, crag_problems)
    system.set_mode(mode)

    if args.question:
        for q in args.question:
            print("#" * 70 + f"\nQ: {q}")
            system.answer(q)
        return

    chat(system)


if __name__ == "__main__":
    main()
