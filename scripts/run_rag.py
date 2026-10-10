"""
Run the standard RAG baseline (and a no-retrieval baseline) on a fixed PubHealth subset.

    python scripts/run_rag.py                  # 200 claims, Qwen2.5-3B
    python scripts/run_rag.py --model 7b       # Qwen2.5-7B in 4-bit
    python scripts/run_rag.py --n 50 --examples 5
    python scripts/run_rag.py --claim "Vitamin C cures the common cold."

Prints accuracy for each setting, shows a few full responses, and saves per-claim results to results/.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crag.data import load_pubhealth, load_textbooks_corpus  # noqa: E402
from crag.generator import generate_answer, load_generator  # noqa: E402
from crag.pipeline import run_rag  # noqa: E402
from crag.retriever import build_index, retrieve_with_scores  # noqa: E402

MODELS = {
    "0.5b": dict(model_name="Qwen/Qwen2.5-0.5B-Instruct"),
    "3b": dict(model_name="Qwen/Qwen2.5-3B-Instruct"),
    "7b": dict(model_name="Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True),
}
INDEX_PATH = ROOT / "indexes" / "contriever-msmarco_textbooks.pt"


def show(claim, gold, index, model, tokenizer, k):
    """
    Print the retrieved passages and the full RAG response for one claim.
    """
    print("\n" + "#" * 70)
    print(f"CLAIM: {claim}" + (f"\nGOLD:  {gold}" if gold else ""))
    print("RETRIEVED:")
    for p, s in retrieve_with_scores(index, claim, k=3):
        print(f"  [{s:.2f}] {p[:150]}...")
    out = run_rag(claim, index, model, tokenizer, k=k)
    print(f"PREDICTED: {out.label} {out.probs}\nRESPONSE:\n{out.response}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="number of PubHealth test claims (stratified, fixed seed)")
    parser.add_argument("--model", choices=MODELS, default="3b")
    parser.add_argument("--k", type=int, default=5, help="passages retrieved per claim")
    parser.add_argument("--examples", type=int, default=3, help="claims to show with a full RAG response")
    parser.add_argument("--claim", action="append", help="answer your own claim instead (repeat for several)")
    args = parser.parse_args()

    index = build_index(load_textbooks_corpus(), cache_path=INDEX_PATH)
    model, tokenizer = load_generator(**MODELS[args.model])

    if args.claim:
        for claim in args.claim:
            show(claim, None, index, model, tokenizer, args.k)
        return

    claims = load_pubhealth(n=args.n)

    rows = []
    start = time.time()
    for r in tqdm(claims.itertuples(), total=len(claims), desc="Claims"):
        none = generate_answer(model, tokenizer, r.claim, [], explain=False)
        rag = run_rag(r.claim, index, model, tokenizer, k=args.k, explain=False)
        rows.append({"claim": r.claim, "gold": r.label,
                     "no_retrieval": none.label, "no_retrieval_p_true": none.probs["true"],
                     "rag": rag.label, "rag_p_true": rag.probs["true"]})
    results = pd.DataFrame(rows)
    elapsed = time.time() - start

    print(f"\n{len(results)} claims ({claims.label.value_counts().to_dict()}), model {args.model}, "
          f"{elapsed:.0f}s ({elapsed / len(results):.2f}s per claim)")
    print(f"  always 'true'   accuracy = {(results.gold == 'true').mean():.3f}")
    for setting in ("no_retrieval", "rag"):
        print(f"  {setting:15s} accuracy = {(results[setting] == results.gold).mean():.3f}")

    for r in claims.head(args.examples).itertuples():
        show(r.claim, r.label, index, model, tokenizer, args.k)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"rag_pubhealth_n{args.n}_{args.model}.csv"
    results.to_csv(path, index=False)
    print(f"\nSaved per-claim results to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
