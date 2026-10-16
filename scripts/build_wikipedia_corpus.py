"""
Build the health-filtered Wikipedia corpus (data/wikipedia_health/) and its Contriever index.

    python scripts/build_wikipedia_corpus.py

Rerun after a crash: finished source files and encoded shards are skipped.
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crag.data import WIKIPEDIA_HEALTH_DIR, load_wikipedia_health_corpus  # noqa: E402
from crag.retriever import DEFAULT_MODEL, _encode, load_retriever  # noqa: E402

REPO = "MedRAG/wikipedia"
INDEX_PATH = ROOT / "indexes" / "contriever-msmarco_wikipedia_health.pt"
SHARD_DIR = ROOT / "indexes" / "wikipedia_health_shards"
SHARD_SIZE = 100_000

TERMS = """
disease diseases disorder disorders syndrome infection infections infectious virus viral bacteria bacterial
pathogen parasite parasitic fungal cancer cancers tumor tumour carcinoma leukemia lymphoma malignant benign
drug drugs medication medications medicine medicines medical medically pharmaceutical pharmacology pharmacy
vaccine vaccines vaccination immunization antibiotic antibiotics antiviral analgesic opioid opioids
clinical clinic hospital hospitals patient patients physician physicians surgeon surgery surgical nurse nursing
symptom symptoms treatment treatments therapy therapies therapeutic diagnosis diagnostic prognosis
chronic acute inflammation inflammatory lesion lesions
epidemic pandemic outbreak epidemiology epidemiological mortality morbidity
health healthcare hygiene sanitation
nutrition nutritional nutrient nutrients diet dietary vitamin vitamins supplement supplements obesity
diabetes diabetic cardiovascular cardiac heart hypertension cholesterol stroke kidney renal liver hepatic
lung lungs pulmonary respiratory asthma blood brain neurological neurodegenerative dementia alzheimer
mental psychiatric psychiatry psychological depression anxiety schizophrenia autism addiction
hormone hormones endocrine insulin enzyme immune immunity antibody antibodies allergy allergic
pregnancy pregnant prenatal abortion contraception contraceptive fertility childbirth
toxic toxicity poison poisoning carcinogen carcinogenic radiation
tobacco smoking nicotine alcohol alcoholism cannabis
anatomy anatomical organ organs tissue muscle bone bones skin
hiv aids malaria tuberculosis influenza flu measles polio ebola zika covid coronavirus sars
medicare medicaid fda cdc nih
""".split()
TERM_PATTERN = re.compile(r"\b(" + "|".join(sorted(set(TERMS), key=len, reverse=True)) + r")\b", re.IGNORECASE)
MEDICAL_TITLE = re.compile(r"\((medicine|disease|drug|medication|anatomy|biology|pharmacology)\)$", re.IGNORECASE)


def is_health_article(title, opening, min_terms):
    """
    Keep an article if its title is tagged as medical, or its title + opening snippet use enough health terms.
    """
    if MEDICAL_TITLE.search(title):
        return True
    return len({m.lower() for m in TERM_PATTERN.findall(f"{title} {opening}")}) >= min_terms


def filter_file(path, min_terms, retries=3):
    """
    Stream one source file and write its health-article snippets to WIKIPEDIA_HEALTH_DIR. Returns the kept count.
    """
    out = WIKIPEDIA_HEALTH_DIR / Path(path).name
    if out.exists():
        with open(out, encoding="utf-8") as f:
            return sum(1 for _ in f)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    for attempt in range(retries):
        try:
            keep, kept = {}, []
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    r = json.loads(line)
                    if r["wiki_id"] not in keep:  # first snippet of an article = its opening
                        keep[r["wiki_id"]] = is_health_article(r["title"], r["content"], min_terms)
                    if keep[r["wiki_id"]]:
                        kept.append(json.dumps({"id": r["id"], "title": r["title"], "contents": r["contents"]},
                                               ensure_ascii=False))
            tmp = out.with_suffix(".tmp")
            tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            tmp.replace(out)
            return len(kept)
        except (requests.RequestException, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(10)


def build_corpus(min_terms, workers):
    WIKIPEDIA_HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    tree = requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/main/chunk", timeout=30).json()
    files = sorted(f["path"] for f in tree if f["path"].endswith(".jsonl"))
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(filter_file, f, min_terms) for f in files]
        kept = sum(f.result() for f in tqdm(as_completed(futures), total=len(futures), desc="Filtering Wikipedia"))
    print(f"{kept:,} passages kept")


@torch.no_grad()
def build_index(batch_size):
    """
    Encode the corpus in resumable shards, then save the index in the format crag.retriever.build_index loads.
    """
    passages = load_wikipedia_health_corpus()
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_retriever(DEFAULT_MODEL)
    n_shards = (len(passages) + SHARD_SIZE - 1) // SHARD_SIZE
    for s in tqdm(range(n_shards), desc="Encoding shards"):
        shard_path = SHARD_DIR / f"{s:04d}.pt"
        texts = passages[s * SHARD_SIZE:(s + 1) * SHARD_SIZE]
        if shard_path.exists() and torch.load(shard_path)["n"] == len(texts):
            continue
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths -> less padding
        emb = _encode(model, tokenizer, [texts[i] for i in order], batch_size=batch_size).cpu()
        unsorted = torch.empty_like(emb)
        unsorted[torch.tensor(order)] = emb
        torch.save({"n": len(texts), "embeddings": unsorted}, shard_path)
    embeddings = torch.cat([torch.load(SHARD_DIR / f"{s:04d}.pt")["embeddings"] for s in range(n_shards)])
    torch.save({"model_name": DEFAULT_MODEL, "passages": passages, "embeddings": embeddings}, INDEX_PATH)
    print(f"saved {INDEX_PATH.relative_to(ROOT)} ({embeddings.shape[0]:,} passages)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-terms", type=int, default=2, help="distinct health terms needed to keep an article")
    parser.add_argument("--workers", type=int, default=4, help="source files downloaded in parallel")
    parser.add_argument("--batch-size", type=int, default=128, help="encoding batch size")
    args = parser.parse_args()
    build_corpus(args.min_terms, args.workers)
    build_index(args.batch_size)


if __name__ == "__main__":
    main()
