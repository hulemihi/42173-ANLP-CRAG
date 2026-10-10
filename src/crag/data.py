"""
PubMedQA loading and train / val / test splits shared by every stage.
"""

import pandas as pd
from datasets import load_dataset

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
