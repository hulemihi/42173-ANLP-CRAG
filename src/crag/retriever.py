"""
Retriever: return the top-k passages for a question from a corpus (e.g. all PubMedQA abstracts).

Dense retrieval with Contriever (Izacard et al., 2022): question and passages are embedded
separately by the same BERT encoder (mean-pooled token embeddings) and ranked by dot product.
facebook/contriever-msmarco (fine-tuned on MS MARCO) is the retriever the CRAG paper used;
facebook/contriever is the unsupervised version.
"""

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModel

DEFAULT_MODEL = "facebook/contriever-msmarco"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_PASSAGE_TOKENS = 512
BATCH_SIZE = 32


@dataclass
class DenseIndex:
    passages: list           # the corpus, list[str]
    embeddings: torch.Tensor  # (len(passages), hidden), on the model's device
    model: AutoModel
    tokenizer: AutoTokenizer


def _mean_pool(token_embeddings, mask):
    """
    Average the token embeddings, ignoring padding -- this is how Contriever builds sentence embeddings.
    """
    mask = mask.unsqueeze(-1).to(token_embeddings.dtype)
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


@torch.no_grad()
def _encode(model, tokenizer, texts, batch_size=BATCH_SIZE, show_progress=False):
    """
    Embed a list of texts. Returns a tensor of shape (len(texts), hidden).
    """
    batches = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm.auto import tqdm
        batches = tqdm(batches, desc="Encoding corpus")
    out = []
    for i in batches:
        enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=MAX_PASSAGE_TOKENS,
                        padding=True, return_tensors="pt").to(model.device)
        with torch.autocast(device_type=model.device.type, enabled=model.device.type == "cuda"):
            hidden = model(**enc).last_hidden_state
        out.append(_mean_pool(hidden, enc.attention_mask).float())
    return torch.cat(out)


def load_retriever(model_name=DEFAULT_MODEL, device=DEVICE):
    """
    Load the Contriever encoder. Returns (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def build_index(corpus, model_name=DEFAULT_MODEL, device=DEVICE, cache_path=None):
    """
    Build a search index over a list of passage strings. Returns the index object.
    If cache_path is given, the passage embeddings are saved there and reloaded on the next call.
    """
    model, tokenizer = load_retriever(model_name, device)
    corpus = list(corpus)

    if cache_path is not None:
        try:
            cached = torch.load(cache_path, map_location=device)
            if cached["model_name"] == model_name and cached["passages"] == corpus:
                return DenseIndex(corpus, cached["embeddings"], model, tokenizer)
        except FileNotFoundError:
            pass

    embeddings = _encode(model, tokenizer, corpus, show_progress=True)
    if cache_path is not None:
        torch.save({"model_name": model_name, "passages": corpus, "embeddings": embeddings.cpu()}, cache_path)
    return DenseIndex(corpus, embeddings, model, tokenizer)


def retrieve_with_scores(index, question, k=5):
    """
    Return the top-k (passage, score) pairs for the question, most relevant first.
    """
    q = _encode(index.model, index.tokenizer, [question])
    scores = (q @ index.embeddings.T)[0]
    top = scores.topk(min(k, len(index.passages)))
    return [(index.passages[i], float(s)) for s, i in zip(top.values.tolist(), top.indices.tolist())]


def retrieve(index, question, k=5):
    """
    Return the top-k passages (list[str]) for the question, most relevant first.
    """
    return [p for p, _ in retrieve_with_scores(index, question, k)]
