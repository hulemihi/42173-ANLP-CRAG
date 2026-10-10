"""
End-to-end CRAG: retrieve -> evaluate -> decide action -> refine / web search -> generate.
"""

from crag import actions
from crag.evaluator import evaluate_relevance
from crag.generator import DEFAULT_TASK, answer_question, generate_answer
from crag.refinement import refine
from crag.retriever import retrieve
from crag.web_search import rewrite_query, web_search


def run_rag(question, index, gen_model, gen_tokenizer, k=5, task=DEFAULT_TASK, explain=True):
    """
    Plain RAG baseline: retrieve top-k and generate. Returns a GeneratorOutput (use .label for scoring).
    """
    passages = retrieve(index, question, k=k)
    return generate_answer(gen_model, gen_tokenizer, question, passages, task=task, explain=explain)


def ask(question, index, gen_model, gen_tokenizer, k=5):
    """
    Plain RAG for a user's free-form question: retrieve top-k and write a cited answer. Returns a QAOutput.
    """
    passages = retrieve(index, question, k=k)
    return answer_question(gen_model, gen_tokenizer, question, passages)


def run_crag(question, index, eval_model, eval_tokenizer, gen_model, gen_tokenizer, k=5, task=DEFAULT_TASK):
    """
    Full CRAG. Returns (GeneratorOutput, action).
    """
    passages = retrieve(index, question, k=k)
    scores = [evaluate_relevance(eval_model, eval_tokenizer, question, p) for p in passages]
    action = actions.decide_action(scores)

    def score_fn(q, text):
        return evaluate_relevance(eval_model, eval_tokenizer, q, text)

    if action == actions.CORRECT:
        knowledge = refine(question, passages, score_fn)
    elif action == actions.INCORRECT:
        knowledge = web_search(rewrite_query(question))
    else:
        knowledge = refine(question, passages, score_fn) + web_search(rewrite_query(question))

    return generate_answer(gen_model, gen_tokenizer, question, knowledge, task=task), action
