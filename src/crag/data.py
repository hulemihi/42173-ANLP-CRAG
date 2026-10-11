"""
Dataset loading shared by every stage.

- PubMedQA (PQA-L): trains the retrieval evaluator only.
- PubHealth: test set for the full pipeline (true / false health claims, as in the CRAG paper).
- MedRAG Textbooks: retrieval corpus (18 medical textbooks, no overlap with PubMedQA).
"""

import pandas as pd
from datasets import load_dataset

PUBHEALTH_NAME = "ImperialCollegeLondon/health_fact"
PUBHEALTH_LABELS = {0: "false", 1: "mixture", 2: "true", 3: "unproven"}
TEXTBOOKS_NAME = "MedRAG/textbooks"

DATASET_NAME = "qiaojin/PubMedQA"
PQA_LABELED = "pqa_labeled"
PQA_ARTIFICIAL = "pqa_artificial"

# Same split as notebooks/pubmedqa_retrieval_evaluator.ipynb, so the saved
# evaluator checkpoint matches. Only the N_TEST questions are unseen by it.
N_TRAIN = 950
N_VAL = 45
N_TEST = 5
SEED = 42


def join_context(example):
    """
    Example['context']['contexts'] is a list of abstract sections -- join into one string.
    """
    ctx = example["context"]
    if isinstance(ctx, dict) and "contexts" in ctx:
        return " ".join(ctx["contexts"])
    return str(ctx)


def load_pubmedqa(subset=PQA_LABELED):
    """
    Load a PubMedQA subset as a DataFrame with columns: question, context, final_decision.
    """
    ds = load_dataset(DATASET_NAME, subset, split="train")
    records = []
    for ex in ds:
        records.append({
            "question": ex["question"],
            "context": join_context(ex),
            "final_decision": ex.get("final_decision", None),
        })
    return pd.DataFrame(records)


def split_pqa_labeled(df):
    """
    Split PQA-L into (train_df, val_df, test_df) exactly as the evaluator was trained.
    """
    shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    train_df = shuffled.iloc[:N_TRAIN].reset_index(drop=True)
    val_df = shuffled.iloc[N_TRAIN:N_TRAIN + N_VAL].reset_index(drop=True)
    test_df = shuffled.iloc[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST].reset_index(drop=True)
    return train_df, val_df, test_df


def load_pubhealth(split="test", binary=True, n=None, seed=SEED):
    """
    Load PubHealth as a DataFrame with columns: claim, label, explanation, main_text.
    binary=True keeps only true / false claims (987 in the test split), as Self-RAG and CRAG evaluate.
    n: keep a fixed random subset of n claims, stratified by label, so everyone runs on the same claims.
    main_text is the fact-checking article -- usable as oracle context.
    """
    # the hub copy is a loading script (unsupported by datasets>=4), so read its parquet conversion
    ds = load_dataset(PUBHEALTH_NAME, revision="refs/convert/parquet", split=split)
    df = ds.to_pandas()
    df = df[df["label"].isin(PUBHEALTH_LABELS)].copy()
    df["label"] = df["label"].map(PUBHEALTH_LABELS)
    if binary:
        df = df[df["label"].isin(["true", "false"])]
    if n is not None and n < len(df):
        frac = n / len(df)
        df = df.groupby("label", group_keys=False).sample(frac=frac, random_state=seed)
    return df[["claim", "label", "explanation", "main_text"]].reset_index(drop=True)


def load_textbooks_corpus():
    """
    Load the MedRAG Textbooks corpus as a list of passage strings (~126k snippets of ~110 words).
    """
    ds = load_dataset(TEXTBOOKS_NAME, revision="refs/convert/parquet", split="train")
    return [c for c in ds["content"] if c and c.strip()]
