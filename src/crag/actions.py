"""
Action trigger: map evaluator scores to a CRAG action.

- CORRECT:   at least one passage scores above UPPER -> use (refined) retrieved passages
- INCORRECT: all passages score below LOWER         -> discard them, use web search
- AMBIGUOUS: otherwise                              -> combine refined passages + web search

TODO: tune UPPER / LOWER on evaluator scores of real retrieved passages.
"""

CORRECT = "correct"
INCORRECT = "incorrect"
AMBIGUOUS = "ambiguous"

UPPER = 0.5   # placeholder
LOWER = -0.5  # placeholder


def decide_action(scores, upper=UPPER, lower=LOWER):
    """
    Given a list of relevance scores (one per retrieved passage), return CORRECT, INCORRECT or AMBIGUOUS.
    """
    raise NotImplementedError
