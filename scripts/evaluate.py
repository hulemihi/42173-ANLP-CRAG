"""
Evaluate no-retrieval, standard RAG and (with --crag) CRAG on a fixed PubHealth subset.

    python scripts/evaluate.py                  # 200 claims, Qwen2.5-3B
    python scripts/evaluate.py --model 7b       # Qwen2.5-7B in 4-bit
    python scripts/evaluate.py --crag           # also run CRAG (needs the actions / refinement / web_search stages)
    python scripts/evaluate.py --n 50 --examples 5
    python scripts/evaluate.py --claim "Vitamin C cures the common cold."
    python scripts/evaluate.py --corpus textbooks  # default is health-filtered Wikipedia

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

from crag.data import load_pubhealth, load_textbooks_corpus, load_wikipedia_health_corpus  # noqa: E402
from crag.generator import predict_label, load_generator  # noqa: E402
from crag.evaluator import load_evaluator  # noqa: E402
from crag.pipeline import check_crag_ready, run_crag, run_rag  # noqa: E402
from crag.retriever import build_index, retrieve_with_scores  # noqa: E402

MODELS = {
    "0.5b": dict(model_name="Qwen/Qwen2.5-0.5B-Instruct"),
    "3b": dict(model_name="Qwen/Qwen2.5-3B-Instruct"),
    "7b": dict(model_name="Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True),
}
CORPORA = {
    "textbooks": (load_textbooks_corpus, ROOT / "indexes" / "contriever-msmarco_textbooks.pt"),
    "wikipedia": (load_wikipedia_health_corpus, ROOT / "indexes" / "contriever-msmarco_wikipedia_health.pt"),
}


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
    parser.add_argument("--corpus", choices=CORPORA, default="wikipedia", help="retrieval corpus")
    parser.add_argument("--k", type=int, default=5, help="passages retrieved per claim")
    parser.add_argument("--examples", type=int, default=3, help="claims to show with a full RAG response")
    parser.add_argument("--claim", action="append", help="answer your own claim instead (repeat for several)")
    parser.add_argument("--crag", action="store_true", help="also run CRAG with the trained retrieval evaluator")
    args = parser.parse_args()

    if args.crag:
        problems = check_crag_ready()
        if problems:
            print("CRAG is not available yet:\n" + "\n".join(f"  - {p}" for p in problems))
            sys.exit(1)

    load_corpus, index_path = CORPORA[args.corpus]
    index = build_index(load_corpus(), cache_path=index_path)
    model, tokenizer = load_generator(**MODELS[args.model])

    if args.claim:
        for claim in args.claim:
            show(claim, None, index, model, tokenizer, args.k)
        return

    claims = load_pubhealth(n=args.n)
    if args.crag:
        eval_model, eval_tokenizer = load_evaluator()

    rows = []
    start = time.time()
    for r in tqdm(claims.itertuples(), total=len(claims), desc="Claims"):
        none = predict_label(model, tokenizer, r.claim, [], explain=False)
        rag = run_rag(r.claim, index, model, tokenizer, k=args.k, explain=False)
        rows.append({"claim": r.claim, "gold": r.label,
                     "no_retrieval": none.label, "no_retrieval_p_true": none.probs["true"],
                     "rag": rag.label, "rag_p_true": rag.probs["true"]})
        if args.crag:
            crag, action = run_crag(r.claim, index, eval_model, eval_tokenizer, model, tokenizer, k=args.k,
                                    explain=False)
            rows[-1].update({"crag": crag.label, "crag_p_true": crag.probs["true"], "crag_action": action})
    results = pd.DataFrame(rows)
    elapsed = time.time() - start

    print(f"\n{len(results)} claims ({claims.label.value_counts().to_dict()}), model {args.model}, "
          f"corpus {args.corpus}, "
          f"{elapsed:.0f}s ({elapsed / len(results):.2f}s per claim)")
    print(f"  always 'true'   accuracy = {(results.gold == 'true').mean():.3f}")
    settings = ("no_retrieval", "rag", "crag") if args.crag else ("no_retrieval", "rag")
    for setting in settings:
        print(f"  {setting:15s} accuracy = {(results[setting] == results.gold).mean():.3f}")
    if args.crag:
        print(f"  CRAG actions: {results.crag_action.value_counts().to_dict()}")

    for r in claims.head(args.examples).itertuples():
        show(r.claim, r.label, index, model, tokenizer, args.k)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{'crag' if args.crag else 'rag'}_pubhealth_n{args.n}_{args.model}_{args.corpus}.csv"
    results.to_csv(path, index=False)
    print(f"\nSaved per-claim results to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
