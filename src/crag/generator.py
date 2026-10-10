"""
Generator: label a PubHealth claim (true / false) or answer a PubMedQA question (yes / no / maybe)
from supporting passages, with the evidence sentences, a short reasoning, and a natural-language response.

Uses an instruction-tuned causal LM (Qwen/Qwen2.5-3B-Instruct by default, fits in 16 GB of VRAM in bf16).
Qwen/Qwen2.5-0.5B-Instruct is faster for development; Qwen/Qwen2.5-7B-Instruct needs load_in_4bit=True.
"""

import re
from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from crag.refinement import split_into_strips

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_CONTEXT_TOKENS = 2048  # Qwen2.5 handles 32k, this just keeps memory and latency in check
BATCH_SIZE = 8
N_EVIDENCE = 3
MIN_EVIDENCE_WORDS = 5
LOW_CONFIDENCE = 0.5

SYSTEM_PROMPT = "You are a careful biomedical research assistant."

DISCLAIMER = (
    "_Note: This response was generated automatically for research purposes and is not medical advice. "
    "Medical questions should always be directed to a qualified medical professional._"
)


@dataclass(frozen=True)
class Task:
    """
    What the generator is asked to do: the labels, prompt wording and response openings.
    openings[label] = (confident, hedged, no_evidence) sentence starting the response.
    """
    labels: tuple
    input_name: str          # "Claim" / "Question"
    instruction: str         # used when context is given
    instruction_no_context: str
    verdict: str             # "... is {label}." -- tells the model the label when asking for reasoning
    openings: dict


PUBHEALTH = Task(
    labels=("true", "false"),
    input_name="Claim",
    instruction="Read the context and decide whether the health claim is true or false. "
                "Answer with exactly one word: true or false.",
    instruction_no_context="Decide whether the health claim is true or false. "
                           "Answer with exactly one word: true or false.",
    verdict="The claim is {label}.",
    openings={
        "true": ("The available evidence supports this claim.",
                 "This claim is probably true, although the evidence is not conclusive.",
                 "This claim is most likely true."),
        "false": ("The available evidence does not support this claim.",
                  "This claim is probably false, although the evidence is not conclusive.",
                  "This claim is most likely false."),
    },
)

PUBMEDQA = Task(
    labels=("yes", "no", "maybe"),
    input_name="Question",
    instruction="Read the context and answer the question with exactly one word: yes, no, or maybe.",
    instruction_no_context="Answer the medical research question with exactly one word: yes, no, or maybe.",
    verdict="The answer is {label}.",
    openings={
        "yes": ("Yes, the available evidence suggests so.",
                "Probably yes, although the evidence is not conclusive.",
                "Most likely yes."),
        "no": ("No, the available evidence does not support this.",
               "Probably not, although the evidence is not conclusive.",
               "Most likely not."),
        "maybe": ("The evidence is mixed, so there is no clear yes or no answer.",
                  "The evidence is mixed, so there is no clear yes or no answer.",
                  "There is no clear yes or no answer."),
    },
)

DEFAULT_TASK = PUBHEALTH


@dataclass
class GeneratorOutput:
    label: str                                   # one of task.labels -- use this for accuracy / F1
    probs: dict                                  # {label: probability}
    evidence: list = field(default_factory=list)  # supporting sentences from the passages
    reasoning: str = ""
    response: str = ""                           # natural answer shown to the user, ends with DISCLAIMER

    @property
    def confidence(self):
        return self.probs[self.label]


def load_generator(model_name=DEFAULT_MODEL, device=DEVICE, load_in_4bit=False):
    """
    Load the generator. Returns (model, tokenizer).
    load_in_4bit needs `bitsandbytes` and `accelerate` (use it for the 7B model).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # so the last position is the next token for every prompt in a batch
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto").to(device)
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


def _chat(tokenizer, user_message):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _answer_prompt(tokenizer, task, question, context):
    if context:
        context = _truncate(tokenizer, context, MAX_CONTEXT_TOKENS)
        message = f"{task.instruction}\n\nContext: {context}\n\n{task.input_name}: {question}"
    else:
        message = f"{task.instruction_no_context}\n\n{task.input_name}: {question}"
    return _chat(tokenizer, message)


def _label_token_ids(tokenizer, labels):
    """
    First token of each label, in lower case and capitalised ("yes" / "Yes"), as {label: [ids]}.
    """
    return {l: sorted({tokenizer(v, add_special_tokens=False).input_ids[0] for v in (l, l.capitalize())})
            for l in labels}


@torch.no_grad()
def _label_probs(model, tokenizer, prompts, labels):
    """
    Probability of each label as the first generated token, for a batch of prompts.
    Returns a tensor of shape (len(prompts), len(labels)).
    """
    label_ids = _label_token_ids(tokenizer, labels)
    out = []
    for i in range(0, len(prompts), BATCH_SIZE):
        enc = tokenizer(prompts[i:i + BATCH_SIZE], padding=True, add_special_tokens=False,
                        return_tensors="pt").to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        # merge "yes" / "Yes" etc., then renormalise over the labels
        scores = torch.stack([torch.logsumexp(logits[:, label_ids[l]], dim=-1) for l in labels], dim=-1)
        out.append(torch.softmax(scores, dim=-1).cpu())
    return torch.cat(out)


def _select_evidence(model, tokenizer, task, question, passages, label, k=N_EVIDENCE):
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

    prompts = [_answer_prompt(tokenizer, task, question, s) for s in sentences]
    probs = _label_probs(model, tokenizer, prompts, task.labels)[:, task.labels.index(label)]
    top = probs.argsort(descending=True)[:k].tolist()
    return [sentences[i] for i in sorted(top)]  # keep original reading order


@torch.no_grad()
def _generate(model, tokenizer, message, max_new_tokens):
    """
    Greedy-decode the model's reply to a single user message.
    """
    enc = tokenizer(_chat(tokenizer, message), add_special_tokens=False, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1,
                         pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def _generate_reasoning(model, tokenizer, task, question, evidence, label):
    verdict = task.verdict.format(label=label)
    if evidence:
        context = _truncate(tokenizer, " ".join(evidence), MAX_CONTEXT_TOKENS)
        message = (f"Evidence: {context}\n\n{task.input_name}: {question}\n\n"
                   f"{verdict} Using the evidence, explain why in one or two sentences.")
    else:
        message = (f"{task.input_name}: {question}\n\n"
                   f"{verdict} Explain why in one or two sentences.")
    reasoning = _generate(model, tokenizer, message, max_new_tokens=100)

    # The opening sentence already states the answer, so drop a leading "Yes," / "False." etc.
    reasoning = re.sub(rf"^({'|'.join(task.labels)})\b[\s,.:;!-]*", "", reasoning, flags=re.IGNORECASE)

    # Small models sometimes just echo the label -- fall back to a plain explanation.
    if len(reasoning.split()) < MIN_EVIDENCE_WORDS:
        if evidence:
            reasoning = f"The most relevant findings in the retrieved sources point towards \"{label}\"."
        else:
            reasoning = f"Based on general knowledge, the model leans towards \"{label}\"."
    if reasoning[-1] not in ".!?":
        reasoning += "."
    return reasoning[0].upper() + reasoning[1:]


def _opening(task, label, confidence, has_evidence):
    confident, hedged, no_evidence = task.openings[label]
    if not has_evidence:
        return no_evidence
    return confident if confidence >= LOW_CONFIDENCE else hedged


def _compose_response(task, label, confidence, evidence, reasoning):
    parts = [f"{_opening(task, label, confidence, bool(evidence))} {reasoning}"]
    if evidence:
        bullets = "\n".join(f"- \"{s}\"" for s in evidence)
        parts.append(f"Key evidence from the retrieved sources:\n{bullets}")
    else:
        parts.append("No supporting documents were provided, so this answer relies on the model's general knowledge only.")
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


def generate_answer(model, tokenizer, question, passages, task=DEFAULT_TASK, explain=True):
    """
    Answer the question (or label the claim) given passages (list[str]; may be empty for no-retrieval).
    Returns a GeneratorOutput with label, probs, evidence, reasoning and a natural response.
    explain=False skips evidence, reasoning and response (one forward pass) -- use it for accuracy runs.
    """
    passages = [p for p in passages if p and p.strip()]
    prompt = _answer_prompt(tokenizer, task, question, " ".join(passages))
    probs = _label_probs(model, tokenizer, [prompt], task.labels)[0]
    label = task.labels[int(probs.argmax())]
    probs = {l: round(float(p), 4) for l, p in zip(task.labels, probs)}
    if not explain:
        return GeneratorOutput(label=label, probs=probs)

    evidence = _select_evidence(model, tokenizer, task, question, passages, label)
    reasoning = _generate_reasoning(model, tokenizer, task, question, evidence, label)
    response = _compose_response(task, label, probs[label], evidence, reasoning)

    return GeneratorOutput(label=label, probs=probs, evidence=evidence, reasoning=reasoning, response=response)


@dataclass
class QAOutput:
    answer: str                                   # free-form answer, cites passages as [1], [2], ...
    sources: list = field(default_factory=list)   # the numbered passages the answer could cite
    response: str = ""                            # answer + sources + DISCLAIMER, shown to the user


_SOURCE_PHRASE = (r"(the |these )?(provided |given |retrieved |available )?"
                  r"(passages?|sources?|context|text|evidence|documents?)( \[\d+\](,? \[\d+\])*)?")


def _tidy_answer(answer):
    """
    Drop openers like "According to the provided passages [1]," that read oddly to a user.
    """
    # at the start of a sentence: "According to the passages, it is ..." -> "It is ..."
    answer = re.sub(rf"(^|(?<=[.!?] ))(based on|according to|from) {_SOURCE_PHRASE},\s*(\w)",
                    lambda m: m.group(m.lastindex).upper(), answer, flags=re.IGNORECASE)
    # at the end of a clause: "..., as stated in the passages." -> "..."
    answer = re.sub(rf",? (according to|as stated in|as mentioned in) {_SOURCE_PHRASE}(?=[.,;!?])", "", answer,
                    flags=re.IGNORECASE)
    return answer[:1].upper() + answer[1:]


def answer_question(model, tokenizer, question, passages, max_new_tokens=250):
    """
    Free-form answer to a user's question from passages (list[str]; may be empty) -- for demos.
    Accuracy on PubHealth is measured with generate_answer, which needs a fixed label set.
    """
    passages = [p for p in passages if p and p.strip()]
    if passages:
        budget = MAX_CONTEXT_TOKENS // len(passages)
        numbered = "\n\n".join(f"[{i}] {_truncate(tokenizer, p, budget)}" for i, p in enumerate(passages, 1))
        message = (f"A member of the public asked the question below. Answer them directly and naturally, using "
                   f"the numbered sources and citing them inline like [1]. Do not mention \"passages\", "
                   f"\"the provided text\" or \"the context\", and do not start with \"According to\". "
                   f"If the sources do not answer the question, say you could not find it in the available "
                   f"sources, then give a brief answer from general medical knowledge. "
                   f"Answer in at most 4 sentences.\n\nSources:\n{numbered}\n\nQuestion: {question}")
    else:
        message = (f"A member of the public asked the question below. Answer them directly and naturally "
                   f"in at most 4 sentences.\n\nQuestion: {question}")
    answer = _tidy_answer(_generate(model, tokenizer, message, max_new_tokens))

    parts = [answer]
    if passages:
        parts.append("Sources:\n" + "\n".join(f"[{i}] {p[:200]}..." for i, p in enumerate(passages, 1)))
    else:
        parts.append("No supporting documents were provided, so this answer relies on the model's general knowledge only.")
    parts.append(DISCLAIMER)
    return QAOutput(answer=answer, sources=passages, response="\n\n".join(parts))
