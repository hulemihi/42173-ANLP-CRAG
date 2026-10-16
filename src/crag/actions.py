"""
Action trigger: map evaluator scores to a CRAG action.

- CORRECT:   at least one passage scores above UPPER -> use (refined) retrieved passages
- INCORRECT: all passages score below LOWER         -> discard them, use web search
- AMBIGUOUS: otherwise                              -> combine refined passages + web search

UPPER / LOWER tuned on PubHealth validation claims with the Wikipedia corpus.
"""

CORRECT = "correct"
INCORRECT = "incorrect"
AMBIGUOUS = "ambiguous"

UPPER = 0.91
LOWER = 0.90


def decide_action(scores, upper=UPPER, lower=LOWER):
    """
    Given a list of relevance scores (one per retrieved passage), return CORRECT, INCORRECT or AMBIGUOUS.
    """
    if not scores:
        return INCORRECT
    if any(s >= upper for s in scores):
        return CORRECT
    elif all(s < lower for s in scores):
        return INCORRECT
    else:
        return AMBIGUOUS
