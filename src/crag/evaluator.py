"""
Retrieval evaluator: T5 encoder + regression head scoring (question, document) relevance in [-1, 1].

Trained in notebooks/pubmedqa_retrieval_evaluator.ipynb.
"""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel

MODEL_NAME = "t5-small"
CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "models" / "t5_relevance_evaluator_pubmedqa.pt"
MAX_LENGTH = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class T5RelevanceScorer(nn.Module):
    """
    T5 encoder + mean pooling + linear head -> score in [-1, 1].
    """

    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        hidden = self.encoder.config.d_model
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return torch.tanh(self.head(pooled)).squeeze(-1)


def load_evaluator(checkpoint_path=CHECKPOINT_PATH, device=DEVICE):
    """
    Load the trained evaluator. Returns (model, tokenizer).
    """
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5RelevanceScorer(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, tokenizer


@torch.no_grad()
def evaluate_relevance(model, tokenizer, question, doc_text, device=DEVICE):
    """
    Return a relevance score in [-1, 1] for a (question, document) pair.
    """
    text = f"question: {question} document: {doc_text}"
    enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt")
    score = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return float(score.item())
