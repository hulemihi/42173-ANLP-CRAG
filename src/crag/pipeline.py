"""
End-to-end RAG and CRAG: retrieve -> (CRAG only: evaluate -> decide action -> refine / web search) -> generate.

- run_rag / run_crag: true / false label for a claim (predict_label) -- used by scripts/evaluate.py for accuracy.
- ask_rag / ask_crag: free-form cited answer for a user (answer_freeform) -- used by scripts/ask.py.
Both CRAG versions share correct_knowledge, so what the user gets is what was evaluated.
"""

from crag import actions
from crag.evaluator import CHECKPOINT_PATH, evaluate_relevance
from crag.generator import DEFAULT_TASK, answer_freeform, predict_label
from crag.refinement import refine
from crag.retriever import retrieve
from crag.web_search import rewrite_query, web_search


def run_rag(question, index, gen_model, gen_tokenizer, k=5, task=DEFAULT_TASK, explain=True):
    """
    Plain RAG baseline: retrieve top-k and generate. Returns a GeneratorOutput (use .label for scoring).
    """
    passages = retrieve(index, question, k=k)
    return predict_label(gen_model, gen_tokenizer, question, passages, task=task, explain=explain)


def check_crag_ready():
    """
    Quick check that every CRAG stage can run, by calling each one on a tiny dummy input.
    Returns a list of problems (empty if CRAG is ready). Note: this makes one real web_search call.
    """
    def check_checkpoint():
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"{CHECKPOINT_PATH.name} not found")

    checks = [
        ("evaluator checkpoint (models/)", check_checkpoint),
        ("actions.decide_action", lambda: actions.decide_action([0.0])),
        ("refinement.refine", lambda: refine("test", ["First sentence. Second sentence."], lambda q, t: 1.0)),
        ("web_search.rewrite_query", lambda: rewrite_query("Vitamin C cures the common cold.")),
        ("web_search.web_search", lambda: web_search("vitamin C common cold", k=1)),
    ]
    problems = []
    for name, check in checks:
        try:
            check()
        except NotImplementedError:
            problems.append(f"{name}: not implemented yet")
        except Exception as e:
            problems.append(f"{name}: {type(e).__name__}: {e}")
    return problems


def correct_knowledge(question, passages, eval_model, eval_tokenizer, gen_model=None, gen_tokenizer=None):
    """
    CRAG's correction step: score the retrieved passages, pick an action, and build the knowledge to generate from.
    Returns (knowledge, action, scores), where knowledge is a list[str].
    """
    def score_fn(q, text):
        return evaluate_relevance(eval_model, eval_tokenizer, q, text)

    scores = [score_fn(question, p) for p in passages]
    action = actions.decide_action(scores)

    if action == actions.CORRECT:
        knowledge = refine(question, passages, score_fn)
    elif action == actions.INCORRECT:
        query = rewrite_query(question, model=gen_model, tokenizer=gen_tokenizer)
        knowledge = web_search(query)
    else:
        query = rewrite_query(question, model=gen_model, tokenizer=gen_tokenizer)
        knowledge = refine(question, passages, score_fn) + web_search(query)
    return knowledge, action, scores


def run_crag(question, index, eval_model, eval_tokenizer, gen_model, gen_tokenizer, k=5, task=DEFAULT_TASK,
             explain=True):
    """
    Full CRAG: retrieve top-k, correct the knowledge, and generate. Returns (GeneratorOutput, action).
    """
    passages = retrieve(index, question, k=k)
    knowledge, action, _ = correct_knowledge(question, passages, eval_model, eval_tokenizer,
                                             gen_model=gen_model, gen_tokenizer=gen_tokenizer)
    return predict_label(gen_model, gen_tokenizer, question, knowledge, task=task, explain=explain), action


def ask_rag(question, index, gen_model, gen_tokenizer, k=5):
    """
    Plain RAG for a user's free-form question: retrieve top-k and write a cited answer. Returns a QAOutput.
    """
    passages = retrieve(index, question, k=k)
    return answer_freeform(gen_model, gen_tokenizer, question, passages)


def ask_crag(question, index, eval_model, eval_tokenizer, gen_model, gen_tokenizer, k=5):
    """
    CRAG for a user's free-form question: same correction step as run_crag, then a cited answer.
    Returns (QAOutput, action, scores).
    """
    passages = retrieve(index, question, k=k)
    knowledge, action, scores = correct_knowledge(question, passages, eval_model, eval_tokenizer,
                                                  gen_model=gen_model, gen_tokenizer=gen_tokenizer)
    return answer_freeform(gen_model, gen_tokenizer, question, knowledge), action, scores
