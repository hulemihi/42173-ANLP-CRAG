"""
Generator: answer a PubMedQA question (yes / no / maybe) from supporting passages.

TODO: implement with an instruction-tuned LM (start with google/flan-t5-base).
"""

LABELS = ("yes", "no", "maybe")


def load_generator(model_name="google/flan-t5-base"):
    """
    Load the generator. Returns (model, tokenizer).
    """
    raise NotImplementedError


def generate_answer(model, tokenizer, question, passages):
    """
    Return one of LABELS for the question given passages (list[str]; may be empty for no-retrieval).
    """
    raise NotImplementedError
