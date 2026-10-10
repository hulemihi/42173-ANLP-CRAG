"""
Knowledge refinement: decompose passages into strips, score each, keep relevant ones, recompose.
"""

import re


def split_into_strips(text, sentences_per_strip=2):
    """
    Split context into strips of 1-2 sentences.
    """
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [" ".join(sents[i:i+sentences_per_strip])
            for i in range(0, len(sents), sentences_per_strip) if sents[i:i+sentences_per_strip]]


def refine(question, passages, score_fn, threshold=0.0, sentences_per_strip=2):
    """
    Return refined passages (list[str]): strips from `passages` whose score_fn(question, strip)
    exceeds `threshold`, in original order.

    TODO: implement (split -> score -> filter -> recompose).
    """
    raise NotImplementedError
