"""
Generator: answer a PubMedQA question (yes / no / maybe) from supporting passages,
with the evidence sentences, a short reasoning, and a natural-language response.

Uses an instruction-tuned seq2seq LM (google/flan-t5-base by default; flan-t5-large
gives noticeably better reasoning if you can afford it).
"""

from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from crag.refinement import split_into_strips

DEFAULT_MODEL = "google/flan-t5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = ("yes", "no", "maybe")
MAX_INPUT_TOKENS = 512
MAX_CONTEXT_TOKENS = 400  # leaves room for the question and instruction
N_EVIDENCE = 3
MIN_EVIDENCE_WORDS = 5
LOW_CONFIDENCE = 0.5

DISCLAIMER = (
    "_Note: This response was generated automatically for research purposes and is not medical advice. "
    "Medical questions should always be directed to a qualified medical professional._"
)


@dataclass
class GeneratorOutput:
    label: str                                   # one of LABELS -- use this for accuracy / F1
    probs: dict                                  # {label: probability}
    evidence: list = field(default_factory=list)  # supporting sentences from the passages
    reasoning: str = ""
    response: str = ""                           # natural answer shown to the user, ends with DISCLAIMER

    @property
    def confidence(self):
        return self.probs[self.label]


def load_generator(model_name=DEFAULT_MODEL, device=DEVICE):
    """
    Load the generator. Returns (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def _truncate(tokenizer, text, max_tokens):
    """
    Cut text to at most max_tokens tokens so the question and instruction are never truncated away.
    """
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def _answer_prompt(tokenizer, question, context):
    if context:
        context = _truncate(tokenizer, context, MAX_CONTEXT_TOKENS)
        return (f"Read the context and answer the question with yes, no, or maybe.\n\n"
                f"Context: {context}\n\nQuestion: {question}\n\nAnswer:")
    return f"Answer the medical research question with yes, no, or maybe.\n\nQuestion: {question}\n\nAnswer:"


@torch.no_grad()
def _label_probs(model, tokenizer, prompts):
    """
    Probability of each label as the first generated token, for a batch of prompts.
    Returns a tensor of shape (len(prompts), len(LABELS)).
    """
    device = model.device
    enc = tokenizer(prompts, truncation=True, max_length=MAX_INPUT_TOKENS,
                    padding=True, return_tensors="pt").to(device)
    start = torch.full((len(prompts), 1), model.config.decoder_start_token_id, device=device)
    logits = model(**enc, decoder_input_ids=start).logits[:, 0, :]
    label_ids = [tokenizer(l, add_special_tokens=False).input_ids[0] for l in LABELS]
    return torch.softmax(logits[:, label_ids], dim=-1).cpu()


def _select_evidence(model, tokenizer, question, passages, label, k=N_EVIDENCE):
    """
    Rank passage sentences by how strongly each one alone supports `label`, and keep the top k.
    """
    sentences = []
    for p in passages:
        for s in split_into_strips(p, sentences_per_strip=1):
            if len(s.split()) >= MIN_EVIDENCE_WORDS and s not in sentences:
                sentences.append(s)
    if not sentences:
        return []

    prompts = [_answer_prompt(tokenizer, question, s) for s in sentences]
    probs = _label_probs(model, tokenizer, prompts)[:, LABELS.index(label)]
    top = probs.argsort(descending=True)[:k].tolist()
    return [sentences[i] for i in sorted(top)]  # keep original reading order


@torch.no_grad()
def _generate_reasoning(model, tokenizer, question, evidence, label):
    if evidence:
        context = _truncate(tokenizer, " ".join(evidence), MAX_CONTEXT_TOKENS)
        prompt = (f"Evidence: {context}\n\nQuestion: {question}\n\n"
                  f"The answer is {label}. Using the evidence, explain why in one or two sentences.")
    else:
        prompt = (f"Question: {question}\n\n"
                  f"The answer is {label}. Explain why in one or two sentences.")

    enc = tokenizer(prompt, truncation=True, max_length=MAX_INPUT_TOKENS, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=80, num_beams=4, no_repeat_ngram_size=3, early_stopping=True)
    reasoning = tokenizer.decode(out[0], skip_special_tokens=True).strip()

    # Small models sometimes just echo the label -- fall back to a plain explanation.
    if len(reasoning.split()) < MIN_EVIDENCE_WORDS:
        if evidence:
            reasoning = f"The most relevant findings in the retrieved studies point towards \"{label}\"."
        else:
            reasoning = f"Based on general knowledge, the model leans towards \"{label}\"."
    if reasoning[-1] not in ".!?":
        reasoning += "."
    return reasoning[0].upper() + reasoning[1:]


def _opening(label, confidence, has_evidence):
    if not has_evidence:
        return {"yes": "Most likely yes.", "no": "Most likely not.",
                "maybe": "There is no clear yes or no answer."}[label]
    if label == "yes":
        return "Yes, the available evidence suggests so." if confidence >= LOW_CONFIDENCE \
            else "Probably yes, although the evidence is not conclusive."
    if label == "no":
        return "No, the available evidence does not support this." if confidence >= LOW_CONFIDENCE \
            else "Probably not, although the evidence is not conclusive."
    return "The evidence is mixed, so there is no clear yes or no answer."


def _compose_response(label, confidence, evidence, reasoning):
    parts = [f"{_opening(label, confidence, bool(evidence))} {reasoning}"]
    if evidence:
        bullets = "\n".join(f"- \"{s}\"" for s in evidence)
        parts.append(f"Key evidence from the retrieved studies:\n{bullets}")
    else:
        parts.append("No supporting documents were provided, so this answer relies on the model's general knowledge only.")
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


def generate_answer(model, tokenizer, question, passages):
    """
    Answer the question given passages (list[str]; may be empty for no-retrieval).
    Returns a GeneratorOutput with label, probs, evidence, reasoning and a natural response.
    """
    passages = [p for p in passages if p and p.strip()]
    prompt = _answer_prompt(tokenizer, question, " ".join(passages))
    probs = _label_probs(model, tokenizer, [prompt])[0]
    label = LABELS[int(probs.argmax())]
    probs = {l: round(float(p), 4) for l, p in zip(LABELS, probs)}

    evidence = _select_evidence(model, tokenizer, question, passages, label)
    reasoning = _generate_reasoning(model, tokenizer, question, evidence, label)
    response = _compose_response(label, probs[label], evidence, reasoning)

    return GeneratorOutput(label=label, probs=probs, evidence=evidence, reasoning=reasoning, response=response)
