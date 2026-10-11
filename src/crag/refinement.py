"""
Knowledge refinement: decompose passages into strips, score each, keep relevant ones, recompose.
"""

import re


def split_into_strips(text, sentences_per_strip=2):
    """
    Split context into strips of 1-2 sentences.
    """
    text = text.strip()
    if not text:
        return []

    sents = re.split(r'(?<=[.!?])\s+', text)
    return [" ".join(sents[i:i+sentences_per_strip])
            for i in range(0, len(sents), sentences_per_strip) if sents[i:i+sentences_per_strip]]

def refine(question, passages, score_fn, threshold=-0.5, sentences_per_strip=2, top_k=5):
    """
    Return the top-k relevant knowledge strips in their original order.
    """
    strips = [
        strip
        for passage in passages
        for strip in split_into_strips(passage, sentences_per_strip)
    ]
    scored_strips = [
        (index, strip, score_fn(question, strip))
        for index, strip in enumerate(strips)
    ]
    relevant_strips = [
        (index, strip, score)
        for index, strip, score in scored_strips
        if score >= threshold
    ]
    selected = sorted(relevant_strips, key=lambda item: item[2], reverse=True)[:max(top_k, 0)]
    selected_indices = {index for index, _, _ in selected}

    return [
        strip
        for index, strip, _ in scored_strips
        if index in selected_indices
    ]